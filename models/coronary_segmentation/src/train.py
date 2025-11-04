import argparse
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from dataset import list_pairs, split_pairs, CoronarySegDataset
from unet import UNet3D
from tqdm import tqdm


def dice_loss(pred, target, eps=1e-6):
    num = 2.0 * (pred * target).sum(dim=(2, 3, 4))
    den = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + eps
    loss = 1.0 - (num + eps) / (den + eps)
    return loss.mean()

@torch.no_grad()
def dice_metric(pred, target, thr=0.5, eps=1e-6):
    p = (pred >= thr).float()
    num = 2.0 * (p * target).sum(dim=(2, 3, 4))
    den = p.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + eps
    return (num / den).mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_roots", nargs="+", default=[
        "data/processed/1-200", "data/processed/201-400"
    ])
    ap.add_argument("--patch", type=int, nargs=3, default=[160, 160, 160])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="models/coronary_segmentation/checkpoints")
    ap.add_argument("--runs_dir", type=str, default="models/coronary_segmentation/runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos_ratio", type=float, default=0.6)
    ap.add_argument("--debug_every", type=int, default=10, help="print/log every N batches")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir); runs_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_pairs(args.data_roots)
    assert len(pairs) > 0, "No image/label pairs found in data_roots."
    train_p, val_p, test_p = split_pairs(pairs, val_frac=0.15, test_frac=0.15, seed=args.seed)

    train_ds = CoronarySegDataset(train_p, tuple(args.patch), args.pos_ratio, train=True)
    val_ds   = CoronarySegDataset(val_p,   tuple(args.patch), args.pos_ratio, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = 0.0
    log_path = runs_dir / f"train_{int(time.time())}.log"
    with open(log_path, "w") as logf:
        logf.write(f"pairs={len(pairs)} train={len(train_ds)} val={len(val_ds)}\n")
        logf.write(f"patch={args.patch} batch={args.batch_size} lr={args.lr}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f"Train | Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for bidx, (vols, gts) in enumerate(loop, start=1):
            vols = vols.to(device, non_blocking=True)
            gts  = gts.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                preds = model(vols)
                loss = dice_loss(preds, gts)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

            if (bidx % args.debug_every) == 0:
                msg_b = f"Epoch {epoch}/{args.epochs} | Batch {bidx}/{len(train_loader)} | loss={loss.item():.6f}"
                tqdm.write(msg_b)
                with open(log_path, "a") as logf:
                    logf.write(msg_b + "\n")

        epoch_loss /= max(1, len(train_loader))

        # ---- Validation ----
        model.eval()
        dices = []
        vloop = tqdm(val_loader, desc=f"Valid | Epoch {epoch}/{args.epochs}", dynamic_ncols=True, leave=False)
        with torch.no_grad():
            for vols, gts in vloop:
                vols = vols.to(device, non_blocking=True)
                gts  = gts.to(device, non_blocking=True)
                preds = model(vols)
                d = dice_metric(preds, gts)
                dices.append(d)
                vloop.set_postfix(dice=f"{d:.4f}")

        val_dice = float(sum(dices) / max(1, len(dices)))

        msg = f"Epoch {epoch:03d} | train_dice_loss={epoch_loss:.4f} | val_dice={val_dice:.4f}"
        print(msg)
        with open(log_path, "a") as logf:
            logf.write(msg + "\n")

        if val_dice > best_val:
            best_val = val_dice
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "val_dice": val_dice,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ✓ Saved best checkpoint (val_dice={val_dice:.4f})")

    print(f"Done. Best val Dice: {best_val:.4f}")

if __name__ == "__main__":
    main()
