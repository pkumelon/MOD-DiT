"""
CogVideoX Attention with Adaptive Mask Generation

集成了attn_mask2.py中的自适应Mask生成功能（函数式实现，支持FlashInfer和SpargeSageAttn backend）
"""

import torch
import logging
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import rearrange
from MoD.mod.attn_mask import (
    SparseAttentionWithMap,
)
from typing import Optional
from diffusers.models.embeddings import apply_rotary_emb
import math
import torch.distributed as dist
import time

logger = logging.getLogger("CogVideoX_Inference")
try:
    from xfuser.core.distributed import get_ulysses_parallel_world_size
    from xfuser.model_executor.layers.usp import _ft_c_input_all_to_all, _ft_c_output_all_to_all
except:
    pass

MAX_TIMING_COUNT = 30
_warmup_time = 0
_warmup_timing_count = 0
_predict_time = 0
_predict_timing_count = 0
_inference_time = 0
_inference_timing_count = 0


def reset_peak_gpu_stats(device: Optional[torch.device] = None):
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.cuda.reset_peak_memory_stats(device.index)

def print_peak_gpu_stats(prefix: str = "", device: Optional[torch.device] = None):
    if not torch.cuda.is_available():
        print(prefix + " GPU not available")
        return
    # 确保所有 kernel 完成
    torch.cuda.synchronize()
    if device is None:
        alloc = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
    else:
        alloc = torch.cuda.max_memory_allocated(device.index)
        reserved = torch.cuda.max_memory_reserved(device.index)
    def to_gib(x): return x / (1024**3)
    print(f"{prefix} peak_alloc={to_gib(alloc):.3f} GiB, peak_reserved={to_gib(reserved):.3f} GiB")

class CogVideoXAttnAdaptiveProcessor2_0:
    """
    自适应稀疏Attention处理器
    
    使用attn_mask2.py的函数式接口在warm up阶段学习attention pattern,
    然后在推理阶段自动生成稀疏mask
    支持FlashInfer和SpargeSageAttn两种backend加速
    """
    
    # 类变量，用于全局配置
    warmup_state = None  
    dense_timestep = 0
    dense_block = 0
    warmup_steps = 12
    attention_mask = None
    
    def __init__(self, layer_idx):
        """
        初始化Attention处理器
        
        参数:
            layer_idx: 当前层索引
        """
        self.layer_idx = layer_idx
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnAdaptiveProcessor2_0 requires PyTorch 2.0")
        
        self.use_sp = False
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        numeral_timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        执行Attention计算
        
        根据timestep和layer_idx决定使用dense还是adaptive sparse attention
        """
        text_seq_length = encoder_hidden_states.size(1)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        
        # QKV线性变换
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # Apply RoPE
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
        
        # 处理sequence parallel
        if self.use_sp:
            query_text = query[:, :, :text_seq_length, :]
            query_video = query[:, :, text_seq_length:, :]
            query_text = _ft_c_input_all_to_all(query_text)
            query_video = _ft_c_input_all_to_all(query_video)
            query = torch.cat([query_text, query_video], dim=-2)
            
            key_text = key[:, :, :text_seq_length, :]
            key_video = key[:, :, text_seq_length:, :]
            key_text = _ft_c_input_all_to_all(key_text)
            key_video = _ft_c_input_all_to_all(key_video)
            key = torch.cat([key_text, key_video], dim=-2)
            
            value_text = value[:, :, :text_seq_length, :]
            value_video = value[:, :, text_seq_length:, :]
            value_text = _ft_c_input_all_to_all(value_text)
            value_video = _ft_c_input_all_to_all(value_video)
            value = torch.cat([value_text, value_video], dim=-2)
        
        global _warmup_time, _predict_time, _inference_time
        global _warmup_timing_count, _predict_timing_count, _inference_timing_count
        # 决定使用哪种attention
        use_dense = (
            timestep is None or 
            numeral_timestep < self.dense_timestep or 
            self.layer_idx < self.dense_block or 
            self.warmup_state is None or
            numeral_timestep < self.warmup_state[0]['warmup_steps'] - 2
        )
        torch.cuda.synchronize()
        start_time = time.time()
        # hidden_states = spas_sage_attn_meansim_topk_cuda(query, key, value, simthreshd1=-0.1, topk=0.5, pvthreshd=15, is_causal=False)
        if use_dense:
            # 使用标准dense attention
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            torch.cuda.synchronize()
            end_time = time.time()
            if _warmup_timing_count < MAX_TIMING_COUNT:
                _warmup_time += end_time - start_time
            if _warmup_timing_count == MAX_TIMING_COUNT:
                logger.info(f"Averge Warmup Time: {_warmup_time / MAX_TIMING_COUNT}")
            _warmup_timing_count += 1
        else:
            # 使用自适应稀疏attention（函数式接口）
            hidden_states_attn = []
            for i in range(batch_size):
                query_i = query[i].unsqueeze(0)  # (1, num_heads, seq_len, head_dim)
                key_i = key[i].unsqueeze(0)      # (1, num_heads, seq_len, head_dim)
                value_i = value[i].unsqueeze(0)  # (1, num_heads, seq_len, head_dim)

                current_step_value = numeral_timestep if isinstance(numeral_timestep, int) else numeral_timestep.item()
                hidden_states_i, mask_info = SparseAttentionWithMap(
                    query=query_i,
                    key=key_i,
                    value=value_i,
                    logger=logger,
                    warmup_state=self.warmup_state[i],
                    current_step=current_step_value,
                    layer_idx=self.layer_idx,
                )
                
                hidden_states_attn.append(hidden_states_i)
            hidden_states = torch.cat(hidden_states_attn, dim=0)
            end_time = time.time()
            if numeral_timestep < self.warmup_state[0]['warmup_steps'] or \
            (numeral_timestep - self.warmup_state[0]['warmup_steps']) % self.warmup_state[0]['predict_T'] == self.warmup_state[0]['predict_T'] - 1:
                if _predict_timing_count < MAX_TIMING_COUNT:
                    _predict_time += end_time - start_time
                if _predict_timing_count == MAX_TIMING_COUNT:
                    logger.info(f"Averge Predict Time: {_predict_time / MAX_TIMING_COUNT}")
                _predict_timing_count += 1
            
            else:
                if _inference_timing_count < MAX_TIMING_COUNT:
                    _inference_time += end_time - start_time
                if _inference_timing_count == MAX_TIMING_COUNT:
                    logger.info(f"Averge Inference Time: {_inference_time / MAX_TIMING_COUNT}")
                _inference_timing_count += 1
        
        # 处理sequence parallel输出
        if self.use_sp:
            try:
                out = hidden_states
                out_text = out[:, :, :get_ulysses_parallel_world_size() * text_seq_length, :]
                out_video = out[:, :, get_ulysses_parallel_world_size() * text_seq_length:, :]
                out_text = _ft_c_output_all_to_all(out_text)
                out_video = _ft_c_output_all_to_all(out_video)
                hidden_states = torch.cat([out_text, out_video], dim=-2)
            except:
                pass
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        
        # Linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        # 分离encoder和decoder的hidden states
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        
        return hidden_states, encoder_hidden_states