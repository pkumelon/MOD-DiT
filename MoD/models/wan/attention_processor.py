from typing import Optional, Tuple
import logging
import time
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import WanAttention, _get_qkv_projections, _get_added_kv_projections
from torch.nn.attention import sdpa_kernel, SDPBackend
from MoD.mod.attn_mask import (
    SparseAttentionWithMap,
)

logger = logging.getLogger("WanVideo_Inference")

MAX_TIMING_COUNT = 30
_warmup_time = 0
_warmup_timing_count = 0
_predict_time = 0
_predict_timing_count = 0
_inference_time = 0
_inference_timing_count = 0
# from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.distributed as dist

try:
    from xfuser.core.distributed import get_ulysses_parallel_world_size
    from xfuser.model_executor.layers.usp import _ft_c_input_all_to_all, _ft_c_output_all_to_all
except:
    pass

class WanAttnAdaptiveProcessor:
    _attention_backend = "flash"
    warmup_state = None  
    dense_timestep = 0
    dense_block = 0
    warmup_steps = 12
    use_cuda = True
    attention_mask = None
    neg = False

    def __init__(self, layer_idx: int):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        self.layer_idx = layer_idx
        self.current_step = 0
        self.use_sp = False

        if dist.is_initialized() and get_ulysses_parallel_world_size() > 1:
            self.use_sp = True

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        global _warmup_time, _predict_time, _inference_time
        global _warmup_timing_count, _predict_timing_count, _inference_timing_count 
        use_dense = ( 
            self.layer_idx < self.dense_block or
            self.warmup_state is None or
            self.current_step < self.warmup_state[0]['warmup_steps'] - 2
        )
        query = query.transpose(1, 2)  # N, Heads, Seq Len, Head Dim
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        torch.cuda.synchronize()
        start_time = time.time()  
        if attn.cross_attention_dim_head is not None or self.warmup_state[0]['sparse_type'] == 'dense': # case for cross attention
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                hidden_states = F.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.0, attn_mask=attention_mask,is_causal=False
                    )
            torch.cuda.synchronize()
            end_time = time.time()
            if _warmup_timing_count < MAX_TIMING_COUNT:
                _warmup_time += end_time - start_time
            if _warmup_timing_count == MAX_TIMING_COUNT:
                logger.info(f"Averge Warmup Time: {_warmup_time / MAX_TIMING_COUNT}")
            _warmup_timing_count += 1

        else: # case for sparse attention
            # 使用自适应稀疏attention（函数式接口）
            current_step_value = self.current_step
            if not self.neg:
                hidden_states, mask_info = SparseAttentionWithMap(
                    query=query,
                    key=key,
                    value=value,
                    warmup_state=self.warmup_state[0],
                    logger=logger,
                    current_step=current_step_value,
                    layer_idx=self.layer_idx,
                    use_cuda=self.use_cuda,
                    use_dense=use_dense,
                )
                self.neg = True
            else:
                hidden_states, mask_info = SparseAttentionWithMap(
                    query=query,
                    key=key,
                    value=value,
                    warmup_state=self.warmup_state[1],
                    logger=logger,
                    current_step=current_step_value,
                    layer_idx=self.layer_idx,
                    use_cuda=self.use_cuda,
                    use_dense=use_dense,
                )
                self.neg = False
            torch.cuda.synchronize()
            end_time = time.time()
            if self.current_step < self.warmup_state[0]['warmup_steps'] or \
            (self.current_step - self.warmup_state[0]['warmup_steps']) % self.warmup_state[0]['predict_T'] == self.warmup_state[0]['predict_T'] - 1:
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
        hidden_states = hidden_states.transpose(1, 2)  # N, Seq Len, Heads, Head Dim
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


