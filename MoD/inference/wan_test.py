import os
import argparse
import torch

from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import UMT5EncoderModel
from MoD.models.wan.utils import replace_wan_attention
from MoD.mod.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="WanVideo inference")

    parser.add_argument(
        "--prompt",
        type=str,
        default="A charming little cat, adorned with a delicate red bow tie, strolls daintily across the emerald green grass.",
    )
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="/data/pretrained_models/")
    parser.add_argument("--output_dir", type=str, default="../results/wan")

    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=69)
    parser.add_argument("--max_seq_length", type=int, default=512)

    # MOD-DiT / sparse attention params
    parser.add_argument("--top_k", type=int, default=350)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=12)
    parser.add_argument("--predict_T", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="wan")
    parser.add_argument("--sparse_type", type=str, default="mod-dit")

    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger = setup_logger(name="WanVideo_Inference", console_output=True)
    logger.info(
        f"Arguments: height={args.height}, width={args.width}, frames={args.frames}, "
        f"top_k={args.top_k}, max_seq_length={args.max_seq_length}"
    )
    logger.info(f"Cache directory exists: {os.path.exists(args.cache_dir)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    vae_dtype = torch.float32
    logger.info(f"Using device={device}, pipe_dtype={pipe_dtype}, vae_dtype={vae_dtype}")

    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32, cache_dir=args.cache_dir, local_files_only=True)
    flow_shift = 5.0
    text_encoder = UMT5EncoderModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, local_files_only=True)
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    transformer = WanTransformer3DModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, local_files_only=True)
    pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", text_encoder=text_encoder, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir, local_files_only=True)
    pipe.scheduler = scheduler

    if args.negative_prompt is None:
        args.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    pipe.to(device)

    replace_wan_attention(
        pipe,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        top_k=args.top_k,
        max_seq_length=args.max_seq_length,
        predict_T=args.predict_T,
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
        guidance_scale=5.0,
        num_inference_steps=50,
        generator=torch.Generator(device=device).manual_seed(42),
    ).frames[0]

    save_root = os.path.join(args.output_dir, f"{args.height}_{args.width}_{args.frames}_topk{args.top_k}_{args.sparse_type}")
    os.makedirs(save_root, exist_ok=True)

    video_path = os.path.join(save_root, "wan_output.mp4")
    export_to_video(video, video_path, fps=15)
    logger.info(f"Video saved to: {video_path}")

if __name__ == "__main__":
    main()