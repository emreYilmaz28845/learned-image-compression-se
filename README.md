# Attention-Augmented Synthesis Transform for Learned Image Compression

CS566 Deep Learning — Spring 2026

This project inserts a Squeeze-and-Excitation (SE) channel attention block into the synthesis transform (decoder) of the Ballé et al. scale hyperprior model and evaluates rate-distortion improvement on the Kodak benchmark.

## Setup

```bash
conda activate learned-image-compression
pip install -r requirements.txt
```

## Project Structure

```
├── README.md
├── train.py                   # Training script for variants A, B, C
├── evaluate.py                # Kodak evaluation (PSNR, MS-SSIM, bpp, BD-rate)
├── plot.py                    # Rate-distortion curve plotting
├── run_all.sh                 # End-to-end experiment pipeline
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── se_hyperprior.py       # SEScaleHyperprior model + weight transfer
│   └── se_block.py            # Squeeze-and-Excitation module
├── utils/
│   ├── __init__.py
│   ├── metrics.py             # PSNR, MS-SSIM, BD-rate
│   └── datasets.py            # CLIC and Kodak dataloaders + download helpers
├── configs/
│   └── baseline.yaml          # Lambda values, lr, batch size, paths
└── experiments/
    └── results/               # Evaluation CSV/JSON outputs and plots
```

## Variants

| Variant | Description |
|---------|-------------|
| A | Pretrained CompressAI baseline (no fine-tuning) |
| B | Baseline fine-tuned for equal steps (controls for continued training) |
| C | Proposed model with SE block in synthesis transform |

## Quick Start

Run the full pipeline:

```bash
bash run_all.sh
```

Or run individual steps:

```bash
# 1. Download datasets
python utils/datasets.py --data-dir data --dataset all

# 2. Train (example: variant C, lambda=0.0067)
python train.py --variant C --lmbda 0.0067 --data-dir data/clic --epochs 100

# 3. Evaluate on Kodak
python evaluate.py --variant C --lmbda 0.0067 \
    --checkpoint checkpoints/variant_C_lmbda_0.0067_mse/checkpoint_best.pth.tar \
    --data-dir data/kodak

# 4. Plot RD curves (after evaluating all variants and lambdas)
python plot.py --results-dir experiments/results --output experiments/plots
```

## Training Details

- **Dataset**: CLIC 2020 (~1800 images), 256x256 random crops
- **Evaluation**: Kodak PhotoCD (24 images, 768x512)
- **Lambda values**: 0.0018, 0.0035, 0.0067, 0.013
- **Optimizer**: Adam, lr=1e-4, ReduceLROnPlateau (factor=0.1, patience=10)
- **Batch size**: 8
- **Distortion**: MSE (primary), MS-SSIM (supplementary via `--distortion msssim`)

## Architecture

The SE block is inserted after the second transposed convolution in the synthesis transform `g_s`:

```
Baseline g_s:
  DeconvT(192→128) → IGDN → DeconvT(128→128) → IGDN → DeconvT(128→128) → IGDN → DeconvT(128→3)

Modified g_s:
  DeconvT(192→128) → IGDN → DeconvT(128→128) → SE(128) → IGDN → DeconvT(128→128) → IGDN → DeconvT(128→3)
```

The SE block adds 2,184 parameters (~0.04% overhead).

## References

1. Ballé et al. (2018). Variational image compression with a scale hyperprior. ICLR 2018.
2. Hu et al. (2018). Squeeze-and-excitation networks. CVPR 2018.
3. CompressAI: https://github.com/InterDigitalInc/CompressAI
