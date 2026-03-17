"""
Training script for learned image compression variants.

Variants:
  A - Pretrained baseline (no training, just save for evaluation)
  B - Fine-tuned baseline without SE block (controls for continued training)
  C - Fine-tuned with SE block (proposed model)

Usage:
  python train.py --variant C --lmbda 0.0067 --data-dir data/clic --epochs 100
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ms_ssim

from dataset import CLICDataset
from models import load_pretrained_with_se, load_pretrained_baseline


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss: L = lambda * D(x, x_hat) + R(y_hat) + R(z_hat)"""

    def __init__(self, lmbda=0.0067, distortion="mse"):
        super().__init__()
        self.lmbda = lmbda
        self.distortion = distortion

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W

        # Rate: bits per pixel from likelihoods
        bpp_loss = sum(
            -torch.log2(likelihoods).sum() / num_pixels
            for likelihoods in output["likelihoods"].values()
        )

        # Distortion
        if self.distortion == "mse":
            distortion = 255.0**2 * nn.functional.mse_loss(output["x_hat"], target)
        else:  # ms-ssim
            distortion = 1.0 - ms_ssim(
                output["x_hat"].clamp(0, 1),
                target,
                data_range=1.0,
                size_average=True,
            )

        loss = self.lmbda * distortion + bpp_loss
        return {
            "loss": loss,
            "bpp": bpp_loss,
            "distortion": distortion,
            "mse": nn.functional.mse_loss(output["x_hat"], target),
        }


def configure_optimizers(model, lr):
    """Separate parameters into main and auxiliary (entropy model) groups."""
    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(model.named_parameters())
    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=1e-3,
    )
    return optimizer, aux_optimizer


def train_one_epoch(model, criterion, train_loader, optimizer, aux_optimizer, device):
    model.train()
    total_loss = 0.0
    total_bpp = 0.0
    total_mse = 0.0
    count = 0

    for i, x in enumerate(train_loader):
        x = x.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out = model(x)
        out["x_hat"] = out["x_hat"].clamp(0, 1)
        losses = criterion(out, x)

        losses["loss"].backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Auxiliary loss for entropy bottleneck
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        total_loss += losses["loss"].item()
        total_bpp += losses["bpp"].item()
        total_mse += losses["mse"].item()
        count += 1

        if (i + 1) % 100 == 0:
            psnr = -10 * math.log10(total_mse / count) if total_mse > 0 else 0
            print(
                f"  Step {i+1}: loss={total_loss/count:.4f} "
                f"bpp={total_bpp/count:.4f} PSNR={psnr:.2f}dB"
            )

    return {
        "loss": total_loss / count,
        "bpp": total_bpp / count,
        "psnr": -10 * math.log10(total_mse / count) if total_mse > 0 else 0,
    }


def validate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_bpp = 0.0
    total_mse = 0.0
    count = 0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            out = model(x)
            out["x_hat"] = out["x_hat"].clamp(0, 1)
            losses = criterion(out, x)

            total_loss += losses["loss"].item()
            total_bpp += losses["bpp"].item()
            total_mse += losses["mse"].item()
            count += 1

    return {
        "loss": total_loss / count,
        "bpp": total_bpp / count,
        "psnr": -10 * math.log10(total_mse / count) if total_mse > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train learned image compression")
    parser.add_argument("--variant", choices=["A", "B", "C"], required=True,
                        help="A=pretrained baseline, B=fine-tuned baseline, C=SE model")
    parser.add_argument("--lmbda", type=float, default=0.0067,
                        help="Rate-distortion trade-off (default: 0.0067)")
    parser.add_argument("--quality", type=int, default=3,
                        help="CompressAI pretrained quality level (default: 3)")
    parser.add_argument("--distortion", choices=["mse", "msssim"], default="mse",
                        help="Distortion metric (default: mse)")
    parser.add_argument("--data-dir", default="data/clic",
                        help="Training data directory")
    parser.add_argument("--val-dir", default=None,
                        help="Validation data directory (default: use 10%% of train)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build run name
    run_name = f"variant_{args.variant}_lmbda_{args.lmbda}_{args.distortion}"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if args.variant == "A":
        print("Variant A: saving pretrained baseline (no training)")
        model = load_pretrained_baseline(quality=args.quality)
        model.to(device)
        model.update()
        torch.save({
            "state_dict": model.state_dict(),
            "variant": "A",
            "quality": args.quality,
        }, save_dir / "checkpoint_best.pth.tar")
        print(f"Saved to {save_dir / 'checkpoint_best.pth.tar'}")
        return

    elif args.variant == "B":
        print("Variant B: fine-tuning baseline without SE block")
        model = load_pretrained_baseline(quality=args.quality)
    elif args.variant == "C":
        print("Variant C: fine-tuning with SE block")
        model = load_pretrained_with_se(quality=args.quality)
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Dataset
    full_dataset = CLICDataset(args.data_dir, patch_size=args.patch_size)
    print(f"Training images: {len(full_dataset)}")

    # Split train/val (90/10)
    val_size = max(1, len(full_dataset) // 10)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Loss, optimizers, scheduler
    criterion = RateDistortionLoss(lmbda=args.lmbda, distortion=args.distortion)
    optimizer, aux_optimizer = configure_optimizers(model, args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, verbose=True
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))

    best_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 20

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, aux_optimizer, device
        )
        val_metrics = validate(model, criterion, val_loader, device)

        scheduler.step(val_metrics["loss"])

        # Log to TensorBoard
        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/bpp", train_metrics["bpp"], epoch)
        writer.add_scalar("train/psnr", train_metrics["psnr"], epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/bpp", val_metrics["bpp"], epoch)
        writer.add_scalar("val/psnr", val_metrics["psnr"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"  Train: loss={train_metrics['loss']:.4f} "
            f"bpp={train_metrics['bpp']:.4f} PSNR={train_metrics['psnr']:.2f}dB"
        )
        print(
            f"  Val:   loss={val_metrics['loss']:.4f} "
            f"bpp={val_metrics['bpp']:.4f} PSNR={val_metrics['psnr']:.2f}dB"
        )

        # Save best model
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "best_loss": best_loss,
                "variant": args.variant,
                "lmbda": args.lmbda,
                "quality": args.quality,
            }, save_dir / "checkpoint_best.pth.tar")
            print(f"  Saved best model (loss={best_loss:.4f})")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "best_loss": best_loss,
                "variant": args.variant,
                "lmbda": args.lmbda,
                "quality": args.quality,
            }, save_dir / f"checkpoint_epoch{epoch}.pth.tar")

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
