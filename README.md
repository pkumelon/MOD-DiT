<div align="center">

# MOD-DiT

### Mixture of Distributions Matters: Dynamic Sparse Attention for Efficient Video Diffusion Transformers

[![arXiv](https://img.shields.io/badge/arXiv-2601.11641-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2601.11641)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](./LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/pkumelon/MOD-DiT?style=flat-square&logo=github)](https://github.com/pkumelon/MOD-DiT/stargazers)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900.svg?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

**[Yuxi Liu](mailto:yuxiliu666@stu.pku.edu.cn)**<sup>1\*</sup> &nbsp;&nbsp;
**[Yipeng Hu](mailto:2301213082@stu.pku.edu.cn)**<sup>1\*</sup> &nbsp;&nbsp;
**[Zekun Zhang](mailto:2023090909020@std.uestc.edu.cn)**<sup>2\*</sup> &nbsp;&nbsp;
**[Kunze Jiang](mailto:kzejiang@mail.ustc.edu.cn)**<sup>3</sup> &nbsp;&nbsp;
**[Kun Yuan](mailto:kunyuan@pku.edu.cn)**<sup>1†</sup>

<sup>1</sup>Peking University &nbsp;&nbsp;
<sup>2</sup>University of Electronic Science and Technology of China &nbsp;&nbsp;
<sup>3</sup>University of Science and Technology of China

<sub>\* Equal contribution &nbsp;&nbsp; † Corresponding author</sub>

**Training-free · Sampling-free · Plug-and-play · Up to 2.05x speedup**

</div>

---

<!-- TODO: Export Figure 1 from the paper and save it to assets/teaser.png -->
<div align="center">
  <img src="assets/teaser.png" alt="MOD-DiT qualitative comparison against other sparse attention methods" width="100%">
  <p><em>MOD-DiT achieves a consistent <b>2.05x</b> speedup on HunyuanVideo while keeping the generated videos almost identical to full attention.</em></p>
</div>

---

## Table of Contents

- [News](#news)
- [Highlights](#highlights)
- [Video Comparison](#video-comparison)
- [Method](#method)
- [Main Results](#main-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

---

## News

- **[2026-01]** Paper released on [arXiv](https://arxiv.org/abs/2601.11641).
- **[2026-07]** Code released, with support for **HunyuanVideo**, **Wan2.1**, and **CogVideoX-v1.5**.

---

## Highlights

Video Diffusion Transformers (vDiTs) are bottlenecked by the quadratic cost of 3D self-attention. Existing sparse attention methods either impose **oversimplified static patterns** (fixed vertical grids, exponential decay) or require **expensive online sampling** to obtain dynamic sparsity.

MOD-DiT resolves this dilemma:

- **Sampling-free dynamic sparsity.** Attention masks are predicted from an analytical linear approximation model instead of repeated online sampling, removing the sampling overhead entirely.
- **Training-free and plug-and-play.** MOD-DiT operates fully online at inference time. No finetuning, no retraining, no auxiliary models.
- **Tri-level adaptivity.** Mask patterns *and* sparsity ratios adapt simultaneously across three dimensions: **input**, **attention head**, and **denoising step**. Notably, the denoising-step dimension is overlooked by most prior dynamic methods.
- **Highest sparsity with the best quality.** MOD-DiT simultaneously attains the highest sparsity (**83.23%** on HunyuanVideo), the best PSNR/LPIPS, and the largest speedup among all compared baselines.
- **Negligible overhead.** All additional operations introduced by MOD-DiT account for only about **12%** of a single full-attention latency, measured on VBench prompts.
- **Hardware-friendly.** Block-wise execution enables localized memory access and leverages GPU warp-level parallelism, integrating cleanly with mainstream sparse-attention APIs.

---

## Video Comparison

Side-by-side comparison between the original **full attention** outputs and **MOD-DiT**, generated from identical prompts and seeds. MOD-DiT preserves subject identity, motion smoothness, and texture detail while cutting inference cost substantially.

<!-- TODO: Place the comparison clips under assets/videos/ using the file names below.
     Recommended: 480p-720p GIF (<10 MB each) for inline playback, plus the original MP4 for full quality.
     GitHub renders GIFs inline, but does not autoplay local MP4 files in README tables. -->

### HunyuanVideo — 117 frames, 768x1280, A100

<table>
<tr>
<td width="50%" align="center"><b>Full Attention</b><br><sub>6978 s &nbsp;|&nbsp; 1.00x</sub></td>
<td width="50%" align="center"><b>MOD-DiT</b><br><sub>3405 s &nbsp;|&nbsp; <b>2.05x</b> &nbsp;|&nbsp; 83.23% sparsity</sub></td>
</tr>
<tr>
<td><img src="assets/videos/hunyuan_full.gif" alt="HunyuanVideo full attention" width="100%"></td>
<td><img src="assets/videos/hunyuan_moddit.gif" alt="HunyuanVideo MOD-DiT" width="100%"></td>
</tr>
</table>

<div align="center">
<sub>
Full-resolution MP4:
<a href="assets/videos/hunyuan_full.mp4">full attention</a> ·
<a href="assets/videos/hunyuan_moddit.mp4">MOD-DiT</a>
</sub>
</div>

### Wan2.1 — 69 frames, 768x1280, A100

<table>
<tr>
<td width="50%" align="center"><b>Full Attention</b><br><sub>3375 s &nbsp;|&nbsp; 1.00x</sub></td>
<td width="50%" align="center"><b>MOD-DiT</b><br><sub>1929 s &nbsp;|&nbsp; <b>1.75x</b> &nbsp;|&nbsp; 81.37% sparsity</sub></td>
</tr>
<tr>
<td><img src="assets/videos/wan_full.gif" alt="Wan2.1 full attention" width="100%"></td>
<td><img src="assets/videos/wan_moddit.gif" alt="Wan2.1 MOD-DiT" width="100%"></td>
</tr>
</table>

<div align="center">
<sub>
Full-resolution MP4:
<a href="assets/videos/wan_full.mp4">full attention</a> ·
<a href="assets/videos/wan_moddit.mp4">MOD-DiT</a>
</sub>
</div>

### CogVideoX-v1.5 — 89 frames, 640x512, A800

<table>
<tr>
<td width="50%" align="center"><b>Full Attention</b><br><sub>987 s &nbsp;|&nbsp; 1.00x</sub></td>
<td width="50%" align="center"><b>MOD-DiT</b><br><sub>542 s &nbsp;|&nbsp; <b>1.82x</b> &nbsp;|&nbsp; 80.10% sparsity</sub></td>
</tr>
<tr>
<td><img src="assets/videos/cogvideox_full.gif" alt="CogVideoX-v1.5 full attention" width="100%"></td>
<td><img src="assets/videos/cogvideox_moddit.gif" alt="CogVideoX-v1.5 MOD-DiT" width="100%"></td>
</tr>
</table>

<div align="center">
<sub>
Full-resolution MP4:
<a href="assets/videos/cogvideox_full.mp4">full attention</a> ·
<a href="assets/videos/cogvideox_moddit.mp4">MOD-DiT</a>
</sub>
</div>

### Reproducing These Comparisons

Generate both variants with the same prompt and seed, changing only `--sparse_type`:

```bash
cd inference

# Full attention reference
python hunyuan_test.py --sparse_type full \
  --prompt "A charming little cat, adorned with a delicate red bow tie, strolls daintily across the emerald green grass." \
  --output_dir ../results/hunyuan_full

# MOD-DiT
python hunyuan_test.py --sparse_type mod-dit \
  --prompt "A charming little cat, adorned with a delicate red bow tie, strolls daintily across the emerald green grass." \
  --output_dir ../results/hunyuan_moddit
```

Then convert the outputs into README-friendly GIFs:

```bash
# Requires ffmpeg
ffmpeg -i results/hunyuan_full/output.mp4 \
  -vf "fps=12,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
  -loop 0 assets/videos/hunyuan_full.gif
```

---

## Method

MOD-DiT is built on a key observation: **attention maps in vDiTs are superpositions of structured patterns that evolve gradually across denoising steps.** Rather than treating each step independently, MOD-DiT models this evolution explicitly through a two-stage process.

<!-- TODO: Export the method overview figure from the paper and save it to assets/framework.png -->
<div align="center">
  <img src="assets/framework.png" alt="MOD-DiT framework overview" width="90%">
</div>

### Stage 1 — Sparsity Map Reconstruction

Using prior information from early denoising steps, MOD-DiT fits an **efficient linear approximation model** that unifies three core structural priors of the attention map. Solving this model yields per-pattern intensity scalars, which are then extrapolated to predict mask patterns for an entire upcoming denoising interval.

<!-- TODO: Export the structural-pattern illustration from the paper and save it to assets/patterns.png -->
<div align="center">
  <img src="assets/patterns.png" alt="Three structural priors composing the attention map" width="80%">
</div>

Given an attention map $S_t \in \mathbb{R}^{n \times n}$ at denoising step $t$, the approximation decomposes it over three structural bases, and the pattern intensities are recovered by solving a small regularized least-squares system:

$$
(M^\top M + \lambda I)\,X = M^\top S_t
$$

Because the design matrix $M$ depends only on the sequence geometry and not on the attention values, $M^\top M$ and its factorization are computed **once** and reused across all heads and all denoising steps. This is implemented as a dedicated CUDA extension in [`mod/cuda_lstsq_kernel.cu`](mod/cuda_lstsq_kernel.cu).

### Stage 2 — Online Block Masking

The predicted masks are then applied through an **online block masking strategy** that maintains historical sparsity information across steps. This removes the need for repetitive sampling operations while keeping the mask responsive to the evolving attention distribution.

### Token-Type-Aware Masking

Modern vDiTs (e.g., CogVideoX, HunyuanVideo) adopt a decoder-only unified attention architecture where text and video tokens are concatenated. To guarantee robust cross-modal alignment at negligible cost, MOD-DiT applies **full attention to text tokens** (typically ~1% of the sequence) and restricts sparsification to video-video attention.

---

## Main Results

All sparse baselines use the **same inference API** to ensure strict fairness. Settings follow the paper: **12 full-attention warm-up steps** and reconstruction interval **t = 10**.

### HunyuanVideo (13B, 117 frames, 768x1280, A100)

| Method | Sparsity | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Subject Consist. ↑ | Imaging Qual. ↑ | Latency ↓ | Speedup ↑ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Attention | 0.00% | – | – | – | 0.9582 | 0.6693 | 6978 s | 1.00x |
| MInference | 67.10% | 18.21 | 0.638 | 0.490 | 0.9300 | 0.6450 | 5286 s | 1.32x |
| Radial Attention | 75.55% | 26.72 | **0.885** | 0.125 | 0.9312 | 0.6467 | 3731 s | 1.87x |
| SVG | 71.22% | 26.44 | 0.861 | 0.170 | 0.8301 | 0.5928 | 3834 s | 1.82x |
| SpargeAttn | 68.00% | 25.43 | 0.842 | 0.195 | 0.9339 | 0.6432 | 4105 s | 1.70x |
| **MOD-DiT (Ours)** | **83.23%** | **27.73** | 0.879 | **0.119** | **0.9398** | **0.6587** | **3405 s** | **2.05x** |

### Wan2.1 (14B, 69 frames, 768x1280, A100)

| Method | Sparsity | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Subject Consist. ↑ | Imaging Qual. ↑ | Latency ↓ | Speedup ↑ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Attention | 0.00% | – | – | – | 0.9623 | 0.6722 | 3375 s | 1.00x |
| MInference | 60.50% | 15.81 | 0.675 | 0.343 | 0.9100 | 0.6550 | 2557 s | 1.32x |
| Radial Attention | 71.33% | 21.57 | 0.818 | 0.167 | 0.9152 | 0.6620 | 2021 s | 1.67x |
| SVG | 69.08% | 20.93 | 0.795 | 0.222 | 0.7986 | 0.6133 | 2109 s | 1.60x |
| SpargeAttn | 50.10% | 18.67 | 0.735 | 0.198 | 0.9033 | 0.6645 | 2296 s | 1.47x |
| **MOD-DiT (Ours)** | **81.37%** | **22.75** | **0.821** | **0.152** | **0.9427** | **0.6674** | **1929 s** | **1.75x** |

### CogVideoX-v1.5 (2B, 89 frames, 640x512, A800)

| Method | Sparsity | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Subject Consist. ↑ | Imaging Qual. ↑ | Latency ↓ | Speedup ↑ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Attention | 0.00% | – | – | – | 0.9230 | 0.6255 | 987 s | 1.00x |
| MInference | 64.90% | 15.01 | 0.601 | 0.334 | 0.8679 | 0.5580 | 696 s | 1.42x |
| Radial Attention | 70.70% | 22.89 | 0.866 | 0.172 | 0.9214 | 0.6167 | 611 s | 1.62x |
| SVG | 75.00% | 21.15 | 0.818 | 0.183 | 0.9158 | 0.5948 | 596 s | 1.65x |
| SpargeAttn | 67.30% | 20.34 | 0.773 | 0.255 | 0.9043 | 0.5966 | 661 s | 1.49x |
| **MOD-DiT (Ours)** | **80.10%** | **25.77** | **0.868** | **0.133** | **0.9266** | **0.6239** | **542 s** | **1.82x** |

### Orthogonal to Step Distillation

MOD-DiT is complementary to timestep-distilled models. Applied on top of **FastWan (8-step)**, it delivers an **additional 1.56x speedup** (327 s → 210 s) with negligible degradation across all VBench quality metrics.

> Quality metrics are reported on [VBench](https://github.com/Vchitect/VBench) prompts. PSNR / SSIM / LPIPS measure fidelity relative to the original full-attention outputs.

---

## Installation

We recommend **CUDA 12.4** with **PyTorch 2.5.1**.

```bash
# 1. Clone the repository together with its submodules
git clone --recursive https://github.com/pkumelon/MOD-DiT.git
cd MOD-DiT

# If you already cloned without --recursive:
# git submodule update --init --recursive

# 2. Create and activate the conda environment
conda create -n mod-dit python==3.12 -y
conda activate mod-dit

# 3. Install PyTorch and Python dependencies
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install flashinfer-python

# 4. Build the sparse attention backends
cd SpargeAttn
pip install -r requirements.txt
python setup.py install
cd ../SageAttention
pip install -r requirements.txt
python setup.py install
cd ..
```

> **Note.** The CUDA extension for the least-squares solver in `mod/` is JIT-compiled on first use via `torch.utils.cpp_extension.load`, so a working `nvcc` matching your PyTorch CUDA version is required. The first run therefore includes a one-time compilation cost.

---

## Quick Start

Each supported model has a dedicated entry point under [`inference/`](inference). Set `--cache_dir` to the directory holding your pretrained weights.

### HunyuanVideo

```bash
cd inference
python hunyuan_test.py \
  --prompt "A charming little cat, adorned with a delicate red bow tie, strolls daintily across the emerald green grass." \
  --cache_dir /path/to/pretrained_models \
  --output_dir ../results/hunyuan \
  --height 768 --width 1280 --frames 117 \
  --sparse_type mod-dit \
  --warmup_steps 12 \
  --top_k 300 \
  --block_size 128 \
  --threshold 1e-4
```

### Wan2.1

```bash
cd inference
python wan_test.py \
  --prompt "A charming little cat, adorned with a delicate red bow tie, strolls daintily across the emerald green grass." \
  --cache_dir /path/to/pretrained_models \
  --output_dir ../results/wan \
  --height 768 --width 1280 --frames 69 \
  --sparse_type mod-dit \
  --warmup_steps 12 \
  --predict_T 10 \
  --top_k 350 \
  --block_size 128
```


### Reproducing the Full-Attention Baseline

```bash
python hunyuan_test.py --sparse_type full   # dense reference
```

---

## Configuration

| Argument | Description | Default |
|:---|:---|:---|
| `--sparse_type` | Attention backend: `mod-dit`, `radial`, `svg`, `spargeattn`, `full` | `mod-dit` |
| `--warmup_steps` | Number of leading full-attention steps used to collect priors | `12` |
| `--predict_T` | Reconstruction interval; masks are refreshed every `predict_T` steps | `10` |
| `--top_k` | Number of structural lines retained when building the mask; **lower means sparser** | `300` (Hunyuan) / `350` (Wan) / `100` (CogVideoX) |
| `--block_size` | Block granularity of the sparse mask, `64` or `128` | `128` |
| `--threshold` | Threshold used when constructing the block sparsity map | `1e-4` |
| `--height`, `--width`, `--frames` | Output video resolution and length | model-specific |
| `--cache_dir` | Directory containing pretrained model weights | model-specific |
| `--output_dir` | Directory for generated videos | `../results/<model>` |

**Tuning guidance.**

- Increasing `block_size` from 64 to 128 reduces latency with negligible impact on quality, and is the recommended setting.
- Larger `predict_T` yields higher speedup but relies more heavily on extrapolation; `predict_T = 10` is the configuration used throughout the paper.
- `warmup_steps` must remain large enough to fit the linear approximation reliably; `12` is used in all reported experiments.

---

## Project Structure

```
MOD-DiT/
├── mod/                          # Core MOD-DiT algorithm
│   ├── attn_mask.py              # Adaptive mask generation and sparse attention dispatch
│   ├── cuda_lstsq.py             # Least-squares solver with cached factorization
│   ├── cuda_lstsq_kernel.cu      # CUDA kernels for the structural least-squares system
│   ├── triton_flash_sparsity.py  # Fused Triton kernel: attention + sparsity map
│   ├── get_radial_mask.py        # Radial Attention baseline
│   └── state.py                  # Warmup / prediction state management
├── inference/                    # Entry points for each supported model
│   ├── hunyuan_test.py
│   ├── wan_test.py
│   └── cogvideo_test.py
├── models/                       # Patched model definitions
├── SageAttention/                # Submodule: quantized attention kernels
├── SpargeAttn/                   # Submodule: block-sparse attention kernels
└── requirements.txt
```

---

## Acknowledgement

This project builds upon several excellent open-source efforts:

- [SageAttention](https://github.com/thu-ml/SageAttention) and [SpargeAttn](https://github.com/thu-ml/SpargeAttn) for high-performance sparse and quantized attention kernels
- [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [Wan2.1](https://github.com/Wan-Video/Wan2.1), and [CogVideoX](https://github.com/THUDM/CogVideo) as base video generation models
- [Diffusers](https://github.com/huggingface/diffusers), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), and [Triton](https://github.com/triton-lang/triton) for infrastructure
- [VBench](https://github.com/Vchitect/VBench) for evaluation protocols

We also thank the authors of Radial Attention, Sparse VideoGen, MInference, and AdaSpa for releasing their implementations, which served as baselines in our experiments.

---

## Citation

If you find MOD-DiT useful or relevant to your research, please cite our paper:

```bibtex
@article{liu2026mixture,
  title={Mixture of Distributions Matters: Dynamic Sparse Attention for Efficient Video Diffusion Transformers},
  author={Liu, Yuxi and Hu, Yipeng and Zhang, Zekun and Jiang, Kunze and Yuan, Kun},
  journal={arXiv preprint arXiv:2601.11641},
  year={2026}
}
```

---

## License

This project is released under the [MIT License](./LICENSE).

Note that the submodules (`SageAttention`, `SpargeAttn`) and the pretrained video generation models are governed by their own respective licenses. Please review them before commercial use.

<div align="center">
<sub>If MOD-DiT helps your work, consider giving the repository a star.</sub>
</div>
