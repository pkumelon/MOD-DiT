import os
import argparse
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from MoD.mod.cogvideox.utils import replace_cogvideox_attention
from MoD.mod.logger import setup_logger
from typing import Dict, List, Tuple, Optional

def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX inference")
 
    parser.add_argument("--prompt", type=str, default="A dog wearing sunglasses, running in a park.")
    parser.add_argument("--cache_dir", type=str, default="/data/pretrained_models/CogVideoX")
    parser.add_argument("--output_dir", type=str, default="../results/cogvideox")
 
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--max_seq_length", type=int, default=128)
 
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
 
    # MOD-DiT
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=12)
    parser.add_argument("--predict_T", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="cogvideox")
    parser.add_argument("--sparse_type", type=str, default="mod-dit")

 
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger(name="CogVideoX_Inference", console_output=True)

    logger.info(
        f"Arguments: height={args.height}, width={args.width}, frames={args.frames}, "
        f"top_k={args.top_k}, max_seq_length={args.max_seq_length}"
    )

    # 设备与 dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"Using device={device}, dtype={dtype}")

    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )

    # 内存优化（按你原逻辑保留）
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    replace_cogvideox_attention(
        pipe,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        top_k=args.top_k,
        max_seq_length=args.max_seq_length,
        model_type=args.model_type,
        block_size=args.block_size,
        warmup_steps=args.warmup_steps,
        use_cuda=True,
        predict_T=args.predict_T,
        sparse_type=args.sparse_type,
    )

    pipe.transformer.config.max_text_seq_length = args.max_seq_length
    pipe.transformer.max_text_seq_length = args.max_seq_length
    pipe.transformer.patch_embed.max_text_seq_length = args.max_seq_length

    pe_device = device
    pipe.transformer.patch_embed.pos_embedding = pipe.transformer.patch_embed._get_positional_embeddings(
        pipe.transformer.config.sample_height,
        pipe.transformer.config.sample_width,
        args.frames,
        device=pe_device
    )

    video = pipe(
        prompt=args.prompt,
        num_videos_per_prompt=1,
        height=args.height,
        width=args.width,
        num_inference_steps=50,
        num_frames=args.frames,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
        max_sequence_length=args.max_seq_length,
    ).frames[0]

    logger.info(f"视频生成完成，prompt: {args.prompt}")

    save_dir = os.path.join(args.output_dir, f"{args.height}_{args.width}")
    os.makedirs(save_dir, exist_ok=True)

    output_path = os.path.join(
        save_dir,
        f"frames_{args.frames}_topk_{args.top_k}_{args.sparse_type}_cogvideo.mp4"
    )
    export_to_video(video, output_path, fps=8)
    logger.info(f"视频已保存到: {output_path}")

if __name__ == "__main__":
    main()