import os
import argparse
import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video
from MoD.mod.logger import setup_logger
from MoD.models.hunyuan.utils import replace_hunyuan_attention

def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo inference")

    parser.add_argument("--prompt", type=str, default="A charming little cat, adorned with a delicate red bow tie, strolls daintily across the emerald green grass.")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="/data/hyp/pretrained_models/")
    parser.add_argument("--output_dir", type=str, default="../results/hunyuan")

    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--max_seq_length", type=int, default=256)


    # MOD-DiT
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=12)
    parser.add_argument("--sparse_type", type=str, default="mod-dit")
    parser.add_argument("--model_type", type=str, default="hunyuan")

    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger = setup_logger(name="HunyuanVideo_Inference", console_output=True)
    logger.info(
        f"Arguments: height={args.height}, width={args.width}, frames={args.frames}, "
        f"top_k={args.top_k}, max_seq_length={args.max_seq_length}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info(f"Using device={device}, dtype={dtype}")

    pipe = HunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=True,
    )

    logger.info("Using model offloading for memory efficiency")
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()

    replace_hunyuan_attention(
        pipe,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        top_k=args.top_k,
        max_seq_length=args.max_seq_length,
        model_type=args.model_type,
        threshold=args.threshold,
        block_size=args.block_size,
        warmup_steps=args.warmup_steps,
        use_cuda=True,
        sparse_type=args.sparse_type,
    )

    logger.info("=" * 20 + " Prompts " + "=" * 20)
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Negative Prompt: {args.negative_prompt}")

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        generator=torch.Generator(device=device).manual_seed(42),
        num_inference_steps=50,
    ).frames[0]

    save_root = os.path.join(args.output_dir, f"{args.height}_{args.width}_{args.frames}_{args.sparse_type}_{args.top_k}")
    os.makedirs(save_root, exist_ok=True)

    # 导出视频
    video_path = os.path.join(save_root, "hunyuan_output.mp4")
    export_to_video(video, video_path, fps=8)
    logger.info(f"Video saved to: {video_path}")

if __name__ == "__main__":
    main()