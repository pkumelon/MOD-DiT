import logging
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import AttentionModuleMixin
from diffusers import WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from MoD.models.wan.attention_processor import WanAttnAdaptiveProcessor
from MoD.models.wan.sparse_pipeline import WanAdaptiveSparsePipeline
from MoD.mod.state import create_adaptive_mask_state
logger = logging.getLogger("WanVideo_Inference")


def replace_wan_attention(
    pipe,
    height,
    width,
    num_frames,
    max_seq_length,
    model_type,
    dense_layers=1,
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
    将 CogVideoX pipeline 的 attention 替换为自适应稀疏 attention

    参数:
        pipe: CogVideoX pipeline 实例
        height: 视频高度
        width: 视频宽度
        num_frames: 视频帧数
        dense_layers: 使用 dense attention 的层数（默认：0，即所有层都用 sparse）
        dense_timesteps: 使用 dense attention 的 timestep 数（默认：0）
        warmup_steps: warm up 阶段的步数（默认：12）
        top_k: 选择最亮的 K 条线（默认：10）
        use_cuda: 是否使用 CUDA 加速（默认：True）
        block_size: block 大小（默认：128）

    返回:
        warmup_state: warmup 状态字典
    """

    # 计算实际的帧数和每帧的 token 数
    num_frames = 1 + num_frames // (pipe.vae_scale_factor_temporal * pipe.transformer.config.patch_size[0])
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    print(f"Mod value (spatial scaling * patch size): {mod_value}")
    print(f"Number of frames after scaling: {num_frames}")
    frame_size = int(height // mod_value) * int(width // mod_value)

    # 计算总的 video token 数量和 sequence length
    video_token_num = frame_size * num_frames

    # 获取 attention head 数量
    num_heads = pipe.transformer.config.num_attention_heads if hasattr(pipe.transformer.config, 'num_attention_heads') else 30

    # 计算总的 head 数量（所有层的 head 数量总和）
    num_layers = len(pipe.transformer.blocks)
    total_heads = num_heads * num_layers

    if sparse_type == 'full':
        dense_timesteps = 50

    # 创建自适应 mask 生成器
    logger.info(f"\n{'='*60}")
    logger.info("Initializing Adaptive Mask Generator for WanVideo")
    logger.info(f"{'='*60}")
    logger.info(f"Video dimensions: {height}x{width}, {num_frames} frames")
    logger.info(f"Frame size (tokens per frame): {frame_size}")
    logger.info(f"Total video tokens: {video_token_num}")
    logger.info(f"Max text sequence length: {max_seq_length}")
    logger.info(f"Number of heads per layer: {num_heads}")
    logger.info(f"Number of layers: {num_layers}")
    logger.info(f"Total heads: {total_heads}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Top-K lines: {top_k}")
    logger.info(f"Dense layers: {dense_layers}")
    logger.info(f"Dense timesteps: {dense_timesteps}")
    logger.info(f"Block size: {block_size}")
    logger.info(f"Sparse Type: {sparse_type}")
    logger.info(f"{'='*60}\n")


    # 配置 Attention 处理器类
    AttnModule = WanAttnAdaptiveProcessor
    AttnModule.dense_block = dense_layers
    AttnModule.dense_timestep = dense_timesteps
    AttnModule.current_step = 0
    AttnModule.warmup_steps = warmup_steps

    # 替换所有 attention 层的 processor
    print("Replacing attention processors...")
    for layer_idx, m in enumerate(pipe.transformer.blocks):
        m.attn1.processor.layer_idx = layer_idx

    # 然后替换所有 Attention 模块的 processor
    replaced_count = 0
    for _, m in pipe.transformer.named_modules():
        if isinstance(m, WanTransformerBlock):
            layer_idx = m.attn1.processor.layer_idx
            m.attn1.set_processor(AttnModule(layer_idx))
            m.attn1.processor.warmup_state = []
            m.attn1.processor.warmup_state.append(create_adaptive_mask_state(
                    model_type=model_type,
                    num_heads=num_heads,
                    video_token_num=video_token_num,
                    text_token_num=max_seq_length,
                    tokens_per_frame=frame_size,
                    num_frames=num_frames,
                    warmup_steps=warmup_steps,
                    top_k=top_k,
                    predict_T=predict_T,
                    threshold=threshold,
                    block_size=block_size,
                    use_cuda=use_cuda,
                    sparse_type=sparse_type,
            ))
            m.attn1.processor.warmup_state.append(create_adaptive_mask_state(
                    model_type=model_type,
                    num_heads=num_heads,
                    video_token_num=video_token_num,
                    text_token_num=max_seq_length,
                    tokens_per_frame=frame_size,
                    num_frames=num_frames,
                    warmup_steps=warmup_steps,
                    top_k=top_k,
                    predict_T=predict_T,
                    threshold=threshold,
                    block_size=block_size,
                    use_cuda=use_cuda,
                    sparse_type=sparse_type,
            ))
            replaced_count += 1

    print(f"Replaced {replaced_count} attention processors\n")
    WanPipeline.__call__ = WanAdaptiveSparsePipeline.__call__

