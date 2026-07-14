# MOD-DiT

### [Paper](https://arxiv.org/abs/2601.11641)

## 🔧Installation

We recommend using CUDA versions 12.4 + Pytorch versions 2.5.1

```bash
# 1. Create and activate conda environment
conda create -n mod-dit python==3.12 -y
conda activate mod-dit

# 2. Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install flashinfer-python

# 3. Install SpargeAttn and SageAttention
cd MOD-DiT
git clone https://github.com/thu-ml/SpargeAttn.git
cd SpargeAttn
pip install -r requirements.txt
python setup.py install
cd ..
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
pip install -r requirements.txt
python setup.py install
```

## 📚Citation

If you find MOD-DiT useful or relevant to your research, please cite our paper:

```bibtex
@article{liu2026mixture,
  title={Mixture of Distributions Matters: Dynamic Sparse Attention for Efficient Video Diffusion Transformers},
  author={Liu, Yuxi and Hu, Yipeng and Zhang, Zekun and Jiang, Kunze and Yuan, Kun},
  journal={arXiv preprint arXiv:2601.11641},
  year={2026}
}
```