"""
Evaluate trained models on the Kodak dataset.

Computes per-image PSNR, MS-SSIM, and bpp, then aggregates results.
Supports BD-rate computation between variants.

Usage:
  python evaluate.py --checkpoint checkpoints/variant_C_lmbda_0.0067_mse/checkpoint_best.pth.tar \
                     --variant C --data-dir data/kodak --output results/
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader

from dataset import KodakDataset
from models import SEScaleHyperprior, load_pretrained_baseline


def load_model(checkpoint_path, variant, quality=3):
    """Load a trained model from checkpoint."""
    if variant == "A":
        model = load_pretrained_baseline(quality=quality)
    elif variant == "B":
        model = load_pretrained_baseline(quality=quality)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
    elif variant == "C":
        quality_to_params = {
            1: (128, 192), 2: (128, 192), 3: (128, 192), 4: (128, 192),
            5: (192, 320), 6: (192, 320), 7: (192, 320), 8: (192, 320),
        }
        N, M = quality_to_params[quality]
        model = SEScaleHyperprior(N=N, M=M)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return model


def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataset. Returns list of per-image metrics."""
    model.eval()
    model.update()
    results = []

    with torch.no_grad():
        for i, (x, (orig_h, orig_w)) in enumerate(data_loader):
            x = x.to(device)
            h, w = orig_h.item(), orig_w.item()

            out = model(x)
            x_hat = out["x_hat"].clamp(0, 1)

            # Crop back to original size
            x_cropped = x[:, :, :h, :w]
            x_hat_cropped = x_hat[:, :, :h, :w]

            # BPP from likelihoods
            num_pixels = h * w
            bpp = sum(
                -torch.log2(lk).sum().item() / num_pixels
                for lk in out["likelihoods"].values()
            )

            # PSNR
            mse = torch.mean((x_cropped - x_hat_cropped) ** 2).item()
            psnr = -10 * math.log10(mse) if mse > 0 else 100.0

            # MS-SSIM
            msssim_val = ms_ssim(
                x_hat_cropped, x_cropped, data_range=1.0, size_average=True
            ).item()

            results.append({
                "image": i + 1,
                "bpp": bpp,
                "psnr": psnr,
                "ms_ssim": msssim_val,
                "ms_ssim_db": -10 * math.log10(1 - msssim_val) if msssim_val < 1 else 100.0,
                "mse": mse,
            })

            print(f"  Image {i+1:2d}: bpp={bpp:.4f}  PSNR={psnr:.2f}dB  MS-SSIM={msssim_val:.4f}")

    return results


def compute_bd_rate(rate_a, psnr_a, rate_b, psnr_b):
    """Compute BD-rate (Bjontegaard Delta Rate) between two RD curves.

    Negative BD-rate means B is better (lower rate at same quality).
    rate_a, psnr_a: reference curve (4 operating points)
    rate_b, psnr_b: test curve (4 operating points)
    """
    # Sort by PSNR
    idx_a = np.argsort(psnr_a)
    idx_b = np.argsort(psnr_b)
    psnr_a = np.array(psnr_a)[idx_a]
    rate_a = np.log(np.array(rate_a)[idx_a])
    psnr_b = np.array(psnr_b)[idx_b]
    rate_b = np.log(np.array(rate_b)[idx_b])

    # Need at least 4 points for cubic polynomial fit
    if len(psnr_a) < 4 or len(psnr_b) < 4:
        print("Warning: BD-rate requires 4 operating points, returning NaN")
        return float("nan")

    # Fit cubic polynomials: rate = f(psnr)
    poly_a = np.polyfit(psnr_a, rate_a, 3)
    poly_b = np.polyfit(psnr_b, rate_b, 3)

    # Integration range
    min_psnr = max(psnr_a.min(), psnr_b.min())
    max_psnr = min(psnr_a.max(), psnr_b.max())

    if min_psnr >= max_psnr:
        return float("nan")

    # Integrate
    p_a = np.polyint(poly_a)
    p_b = np.polyint(poly_b)

    int_a = np.polyval(p_a, max_psnr) - np.polyval(p_a, min_psnr)
    int_b = np.polyval(p_b, max_psnr) - np.polyval(p_b, min_psnr)

    avg_diff = (int_b - int_a) / (max_psnr - min_psnr)
    bd_rate = (np.exp(avg_diff) - 1) * 100

    return bd_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Kodak dataset")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (not needed for variant A)")
    parser.add_argument("--variant", choices=["A", "B", "C"], required=True)
    parser.add_argument("--quality", type=int, default=3)
    parser.add_argument("--lmbda", type=float, default=None,
                        help="Lambda value (for labeling results)")
    parser.add_argument("--data-dir", default="data/kodak")
    parser.add_argument("--output", default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading variant {args.variant}...")
    model = load_model(args.checkpoint, args.variant, args.quality)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Load dataset
    dataset = KodakDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(dataset)} images...")

    # Evaluate
    results = evaluate_model(model, loader, device)

    # Aggregate
    avg_bpp = np.mean([r["bpp"] for r in results])
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_msssim = np.mean([r["ms_ssim"] for r in results])
    avg_msssim_db = np.mean([r["ms_ssim_db"] for r in results])

    print(f"\n--- Averages ---")
    print(f"  BPP:      {avg_bpp:.4f}")
    print(f"  PSNR:     {avg_psnr:.2f} dB")
    print(f"  MS-SSIM:  {avg_msssim:.4f}")
    print(f"  MS-SSIM (dB): {avg_msssim_db:.2f}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    lmbda_str = f"{args.lmbda}" if args.lmbda else "unknown"
    result_file = output_dir / f"variant_{args.variant}_lmbda_{lmbda_str}.csv"

    with open(result_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "bpp", "psnr", "ms_ssim", "ms_ssim_db", "mse"])
        writer.writeheader()
        writer.writerows(results)

    # Save summary
    summary = {
        "variant": args.variant,
        "lmbda": args.lmbda,
        "quality": args.quality,
        "avg_bpp": avg_bpp,
        "avg_psnr": avg_psnr,
        "avg_ms_ssim": avg_msssim,
        "avg_ms_ssim_db": avg_msssim_db,
        "total_params": total_params,
    }
    summary_file = output_dir / f"variant_{args.variant}_lmbda_{lmbda_str}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {result_file}")
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
