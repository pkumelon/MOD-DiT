"""
CogVideoX Inference with Adaptive Mask Generation

初始化和配置自适应稀疏attention（使用attn_mask2.py的函数式接口）
"""

import torch
import logging
from diffusers import CogVideoXPipeline
from diffusers.models.attention_processor import Attention
from MoD.models.cogvideox.attention_processor import CogVideoXAttnAdaptiveProcessor2_0
from MoD.models.cogvideox.sparse_pipeline import CogVideoXAdaptiveSparsePipeline
from MoD.mod.state import create_adaptive_mask_state
from MoD.mod.utils import reset_peak_gpu_stats, print_gpu_stats
from typing import Optional

# 获取 logger
logger = logging.getLogger("CogVideoX_Inference")

def replace_cogvideox_attention(
    pipe,
    height,
    width,
    num_frames,
    max_seq_length,
    model_type='cogvideox',
    dense_layers=0,
    dense_timesteps=0,
    warmup_steps=12,
    top_k=10,
    predict_T=10,
    threshold=1.5e-4,
    use_cuda=True,
    block_size=128,
    sparse_type='mod-dit',
):
    """
    将CogVideoX pipeline的attention替换为自适应稀疏attention
    
    参数:
        pipe: CogVideoX pipeline实例
        height: 视频高度
        width: 视频宽度
        num_frames: 视频帧数
        max_seq_length: 文本序列总长度
        model_type: 模型类型
        dense_layers: 使用dense attention的层数（默认：0，即所有层都用sparse）
        dense_timesteps: 使用dense attention的timestep数（默认：0）
        warmup_steps: warm up阶段的步数（默认：12）
        top_k: 选择最亮的K条线（默认：10）
        use_cuda: 是否使用CUDA加速（默认：True）
        block_size: block大小，用于backend（默认：128）
    
    返回:
        None
    """
    
    # 计算压缩后的帧数和每帧的token数
    compressed_num_frames = 1 + (num_frames - 1) // (pipe.vae_scale_factor_temporal)
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size
    frame_size = int(height // mod_value) * int(width // mod_value)
    
    # 计算总的video token数量和sequence length
    video_token_num = frame_size * compressed_num_frames
    
    # 获取attention head数量
    num_heads = pipe.transformer.config.num_attention_heads if hasattr(pipe.transformer.config, 'num_attention_heads') else 30
    
    # 计算总的head数量（所有层的head数量总和）
    num_layers = len(pipe.transformer.transformer_blocks)
    total_heads = num_heads * num_layers
    
    if sparse_type == 'full':
        dense_timesteps = 50
    # 创建自适应mask生成器
    logger.info(f"\n{'='*60}")
    logger.info("Initializing Adaptive Mask Generator for CogVideoX")
    logger.info(f"{'='*60}")
    logger.info(f"原始Video dimensions: {height}x{width}, {num_frames} frames")
    logger.info(f"Frame size (tokens per frame): {frame_size}")
    logger.info(f"Total video tokens: {video_token_num}")
    logger.info(f"Max text sequence length: {max_seq_length}")
    logger.info(f"Number of heads per layer: {num_heads}")
    logger.info(f"Number of layers: {num_layers}")
    logger.info(f"Total heads: {total_heads}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Top-K lines: {top_k}")
    logger.info(f"Predict period: {predict_T}")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Dense layers: {dense_layers}")
    logger.info(f"Dense timesteps: {dense_timesteps}")
    logger.info(f"Block size: {block_size}")
    logger.info(f"Sparse Type: {sparse_type}")
    logger.info(f"{'='*60}\n")
    
    
    # 配置Attention处理器类
    AttnModule = CogVideoXAttnAdaptiveProcessor2_0
    AttnModule.dense_block = dense_layers
    AttnModule.dense_timestep = dense_timesteps
    AttnModule.warmup_steps = warmup_steps
    
    # 替换所有attention层的processor
    logger.info("Replacing attention processors...")
    # 首先设置每个transformer block的layer_idx
    for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
        if hasattr(m, 'attn1') and hasattr(m.attn1, 'processor'):
            m.attn1.processor.layer_idx = layer_idx
    # 然后替换所有Attention模块的processor
    replaced_count = 0
    for _, m in pipe.transformer.named_modules():
        if isinstance(m, Attention) and hasattr(m.processor, "layer_idx"):
            layer_idx = m.processor.layer_idx
            m.set_processor(AttnModule(layer_idx))
            # 初始化warmup_state
            m.processor.warmup_state = []
            m.processor.warmup_state.append(create_adaptive_mask_state(
                    model_type=model_type,
                    num_heads=num_heads,
                    video_token_num=video_token_num,
                    text_token_num=max_seq_length,
                    tokens_per_frame=frame_size,
                    num_frames=compressed_num_frames,
                    warmup_steps=warmup_steps,
                    top_k=top_k,
                    predict_T=predict_T,
                    threshold=threshold,
                    block_size=block_size,
                    use_cuda=use_cuda,
                    sparse_type=sparse_type,
                )
            )
            m.processor.warmup_state.append(create_adaptive_mask_state(
                    model_type=model_type,
                    num_heads=num_heads,
                    video_token_num=video_token_num,
                    text_token_num=max_seq_length,
                    tokens_per_frame=frame_size,
                    num_frames=compressed_num_frames,
                    warmup_steps=warmup_steps,
                    top_k=top_k,
                    predict_T=predict_T,
                    threshold=threshold,
                    block_size=block_size,
                    use_cuda=use_cuda,
                    sparse_type=sparse_type,
                )
            )
            replaced_count += 1
            reset_peak_gpu_stats()
            print_gpu_stats(tag="After setting layer_idx:", device=torch.device("cuda"),logger=logger)
    logger.info(f"Replaced {replaced_count} attention processors\n")
    CogVideoXPipeline.__call__ = CogVideoXAdaptiveSparsePipeline.__call__
    
