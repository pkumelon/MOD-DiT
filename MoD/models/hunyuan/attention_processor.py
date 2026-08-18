import logging
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from MoD.mod.attn_mask import (
    SparseAttentionWithMap,
)
from MoD.mod.utils import reset_peak_gpu_stats, print_gpu_stats
from typing import Optional
from diffusers.models.embeddings import apply_rotary_emb
from torch.nn.attention import sdpa_kernel, SDPBackend
import time
logger = logging.getLogger("HunyuanVideo_Inference")

MAX_TIMING_COUNT = 30
_warmup_time = 0
_warmup_timing_count = 0
_predict_time = 0
_predict_timing_count = 0
_inference_time = 0
_inference_timing_count = 0


class HunyuanVideoAttnAdaptiveProcessor2_0:
    # 类变量（所有实例共享的配置）
    warmup_state = None
    dense_timestep = 0
    dense_block = 0
    warmup_steps = 12
    attention_mask = None

    def __init__(self, layer_idx):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

        self.layer_idx = layer_idx
        self.current_step = 0  # 实例变量 - 每个 layer 独立计数

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention
        # 决定使用哪种attention
        use_dense = ( 
            self.current_step < self.dense_timestep or
            self.layer_idx < self.dense_block or
            self.warmup_state is None or
            self.current_step < self.warmup_state['warmup_steps'] - 2
        )
        global _warmup_time, _predict_time, _inference_time
        global _warmup_timing_count, _predict_timing_count, _inference_timing_count
        torch.cuda.synchronize()
        start_time = time.time()
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
            pre_defined_mask = attention_mask[0, 0].expand(query.shape[2], query.shape[2]).contiguous()
            current_step_value = self.current_step
            hidden_states, mask_info = SparseAttentionWithMap(
                query=query,
                key=key,
                value=value,
                pre_defined_mask=pre_defined_mask,
                warmup_state=self.warmup_state,
                logger=logger,
                current_step=current_step_value,
                layer_idx=self.layer_idx
            )
            torch.cuda.synchronize()
            end_time = time.time()
            if self.current_step < self.warmup_state['warmup_steps'] or \
            (self.current_step - self.warmup_state['warmup_steps']) % self.warmup_state['predict_T'] == self.warmup_state['predict_T'] - 1:
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
                
            

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        self.current_step += 1
        return hidden_states, encoder_hidden_states