from typing import Dict

def init_warmup_state() -> Dict:
    """
    初始化warmup状态
    
    参数:
        num_heads: attention head的数量
        model_type: 模型类型
        warmup_steps: warmup的步数
    
    返回:
        warmup状态字典
    """
    return {
        'features_history': None,  # {head_idx: List[Dict]}
        'current_steps': 0,  # {head_idx: int}
        'prev_full_maps': None,  # {head_idx: Tensor}
        'prediction_features': {},  # {head_idx: Dict}
        'mask': None,
        'sparse_id': None,
    }

def create_adaptive_mask_state(
    model_type: str,
    num_heads: int,
    video_token_num: int,
    text_token_num: int,
    tokens_per_frame: int,
    num_frames: int,
    warmup_steps: int = 12,
    top_k: int = 10,
    predict_T: int = 5,
    threshold: float = 1.5e-4,
    block_size: int = 128,
    use_cuda: bool = True,
    sparse_type: str = 'mod-dit',
) -> Dict:
    """
    创建自适应mask状态的便捷函数
    """
    # 算法中间变量
    state = init_warmup_state()

    # 模型相关参数
    state['model_type'] = model_type
    state['num_heads'] = num_heads

    # 输入相关参数
    state['video_token_num'] = video_token_num
    state['text_token_num'] = text_token_num
    state['blocks_per_frame'] = tokens_per_frame // block_size
    state['frame_size'] = tokens_per_frame
    state['num_frames'] = num_frames
    
    # 算法相关参数
    state['warmup_steps'] = warmup_steps
    state['top_k'] = top_k
    state['threshold'] = float(threshold)  # Host scalar avoids a GPU sync before every map replay.
    state['block_size'] = block_size
    state['predict_T'] = predict_T
    state['use_cuda'] = use_cuda

    if sparse_type not in {'full', 'mod-dit', 'radial', 'svg', 'spargeattn'}:
        raise ValueError(f"{sparse_type} not support , only support [full mod-dit, radial, svg, spargeattn]")
    else:
        state['sparse_type'] = sparse_type
    return state

def is_warmup_complete(warmup_state: Dict) -> bool:
    """检查warmup是否完成"""
    return warmup_state['current_steps'] >= warmup_state['warmup_steps']