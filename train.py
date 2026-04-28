from email import parser
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision import transforms, models

class NIHChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_cols, transform=None):
        df = df.reset_index(drop=True)
        if "img_path" not in df.columns:
            raise ValueError("df must contain an 'img_path' column with full paths to images.")

        self.img_paths = df["img_path"].astype(str).tolist()
        self.labels = df[label_cols].to_numpy(dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.from_numpy(self.labels[idx])  # float32 multi-hot
        return img, labels


class FocalLossWithLogits(nn.Module):
    """
    Multi-label focal loss (sigmoid + BCE) for logits.
    gamma: how much to focus on hard examples (common: 1.5-2.0)
    alpha: optional class balancing. Can be:
           - None (no alpha)
           - float in [0,1] (same alpha for all classes)
           - Tensor [C] for per-class alpha
    pos_weight: like BCEWithLogitsLoss pos_weight (Tensor [C]) for imbalance
    """
    def __init__(self, gamma=1.5, alpha=None, pos_weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()

        # BCE per element (no reduction)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )

        # p_t = prob of the true class for each label
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        focal_factor = (1 - p_t).pow(self.gamma)

        loss = focal_factor * bce

        # Optional alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # alpha as Tensor [C]
                alpha = self.alpha.to(logits.device)
                alpha_t = alpha[None, :] * targets + (1 - alpha[None, :]) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def read_split_file(path):
    with open(path, "r") as f:
        return {line.strip() for line in f.readlines()}

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, epoch, epochs, accum_steps = 1):
    model.train()
    running_loss = 0.0
    n = 0

    t0 = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Train {epoch}/{epochs}", leave=False)
    for step, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss = loss / accum_steps  # key: normalize

        scaler.scale(loss).backward()

        # only step every accum_steps (or at the end)
        do_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(loader))
        if do_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = imgs.size(0)
        # log the *true* per-sample loss (undo the division for reporting)
        running_loss += (loss.detach().item() * accum_steps) * bs
        n += bs

        pbar.set_postfix(train_loss=running_loss / max(n, 1))

    dt = time.perf_counter() - t0
    return (running_loss / max(n, 1)), dt

def compute_auc_map(logits, targets, label_cols):
    """
    logits/targets: numpy arrays shape [N, C]
    Returns dict with per-class AUROC/mAP and macro averages.
    """
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    aurocs = []
    maps = []
    per_class = {}

    for i, name in enumerate(label_cols):
        y_true = targets[:, i]
        y_prob = probs[:, i]

        # Need both classes present to compute AUROC/AP
        if y_true.min() == y_true.max():
            per_class[name] = {"auroc": None, "map": None}
            continue

        au = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        per_class[name] = {"auroc": float(au), "map": float(ap)}
        aurocs.append(au)
        maps.append(ap)

    macro_auroc = float(np.mean(aurocs)) if len(aurocs) else None
    macro_map = float(np.mean(maps)) if len(maps) else None

    return {
        "macro_auroc": macro_auroc,
        "macro_map": macro_map,
        "per_class": per_class,
    }

def micro_f1_from_probs(probs, targets, thresholds):
    """
    probs/targets: numpy [N, C]
    thresholds: float or numpy [C]
    """
    probs = np.asarray(probs)
    targets = np.asarray(targets).astype(np.int32)

    if np.isscalar(thresholds):
        thr = np.full((probs.shape[1],), float(thresholds), dtype=np.float32)
    else:
        thr = np.asarray(thresholds, dtype=np.float32)

    preds = (probs >= thr[None, :]).astype(np.int32)

    tp = (preds & targets).sum()
    fp = (preds & (1 - targets)).sum()
    fn = ((1 - preds) & targets).sum()

    denom = (2 * tp + fp + fn)
    return float((2 * tp / denom) if denom > 0 else 0.0)

def tune_thresholds_per_class(logits, targets, label_cols, grid=None):
    """
    Tune one threshold per class on validation set to maximize per-class F1.
    Returns thresholds array [C].
    """
    # logits/targets might be torch tensors (your collect function returns torch)
    if torch.is_tensor(logits):
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    else:
        logits = np.asarray(logits)
        probs = 1 / (1 + np.exp(-logits))

    if torch.is_tensor(targets):
        t = targets.detach().cpu().numpy().astype(np.int32)
    else:
        t = np.asarray(targets).astype(np.int32)

    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    C = probs.shape[1]
    best_thr = np.full((C,), 0.5, dtype=np.float32)

    for i in range(C):
        y = t[:, i]
        if y.min() == y.max():
            best_thr[i] = 0.5
            continue

        best_f1 = -1.0
        best_t = 0.5

        for th in grid:
            p = (probs[:, i] >= th).astype(np.int32)
            tp = (p & y).sum()
            fp = (p & (1 - y)).sum()
            fn = ((1 - p) & y).sum()
            denom = (2 * tp + fp + fn)
            f1 = (2 * tp / denom) if denom > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_t = th

        best_thr[i] = best_t

    val_micro = micro_f1_from_probs(probs, t, best_thr)
    return best_thr, float(val_micro)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, epoch=None, epochs=None, threshold=0.5):
    model.eval()
    running_loss = 0.0
    n = 0

    tp = fp = fn = 0

    desc = "Eval" if (epoch is None or epochs is None) else f"Val {epoch}/{epochs}"
    pbar = tqdm(loader, desc=desc, leave=False)

    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)

        bs = imgs.size(0)
        running_loss += loss.detach().item() * bs
        n += bs

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).to(torch.int32)
        t = labels.to(torch.int32)

        tp += (preds & t).sum().item()
        fp += (preds & (1 - t)).sum().item()
        fn += ((1 - preds) & t).sum().item()

        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        pbar.set_postfix(val_loss=running_loss / max(n, 1), micro_f1=f1)

    val_loss = running_loss / max(n, 1)
    denom = (2 * tp + fp + fn)
    micro_f1 = (2 * tp / denom) if denom > 0 else 0.0
    return val_loss, micro_f1

@torch.no_grad()
def collect_logits_targets(model, loader, device):
    """Collect all logits + targets for threshold tuning (runs once after training)."""
    model.eval()
    all_logits = []
    all_targets = []

    for imgs, labels in tqdm(loader, desc="Collect (val)", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)

        all_logits.append(logits.float().cpu())
        all_targets.append(labels.float().cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)

def micro_f1_from_logits_cpu(logits_cpu, targets_cpu, threshold=0.5):
    probs = torch.sigmoid(logits_cpu)
    preds = (probs >= threshold).to(torch.int32)
    t = targets_cpu.to(torch.int32)

    tp = (preds & t).sum().item()
    fp = (preds & (1 - t)).sum().item()
    fn = ((1 - preds) & t).sum().item()

    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


def main():
    # args parsing and setup for training loop
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="train_top5.csv")
    parser.add_argument("--val_csv", type=str, default="val_top5.csv")
    parser.add_argument("--test_csv", type=str, default="test_top5.csv")
    parser.add_argument("--labels_file", type=str, default="top5_labels.txt")
    parser.add_argument("--img_root", type=str, default="NIH dataset images")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--best_ever", type=str, default="runs/best_ever.pt")
    parser.add_argument("--accum_steps", type=int, default=1)

    args = parser.parse_args()

    seed_everything(args.seed)

    img_root = Path(args.img_root)
    if not img_root.exists():
        raise FileNotFoundError(f"img_root not found: {img_root.resolve()}")
    
    # Load already-prepared TOP-5 splits (created by top_5.py)
    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)
    test_df  = pd.read_csv(args.test_csv)

    # Load label order (guarantees consistent class indexing)
    label_cols = pd.read_csv(args.labels_file, header=None)[0].tolist()
    if len(label_cols) != 5:
        raise ValueError(f"Expected 5 labels, got {len(label_cols)} from {args.labels_file}: {label_cols}")

    # Sanity checks
    for name, df_ in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "img_path" not in df_.columns:
            raise ValueError(f"{name} csv must contain 'img_path' (full path).")
        missing = [c for c in label_cols if c not in df_.columns]
        if missing:
            raise ValueError(f"{name} csv missing label columns: {missing}")
        
    # Transforms
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # NOTE: removed RandomHorizontalFlip (laterality matters in chest X-rays)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)), # random crop + resize
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # train/val/test datasets
    train_ds = NIHChestXrayDataset(train_df, label_cols, transform=train_tf)
    val_ds   = NIHChestXrayDataset(val_df,   label_cols, transform=val_tf)
    test_ds  = NIHChestXrayDataset(test_df,  label_cols, transform=val_tf)

    # device + backend perf flags, print info to make sure everything is good
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # dataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
    )

    # Model: DenseNet121 with modified classifier
    num_classes = len(label_cols)
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # Class imbalance handling: pos_weight for each class
    pos = train_df[label_cols].sum(axis=0).values.astype(np.float32)
    neg = (len(train_df) - pos).astype(np.float32)
    pos_weight = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32, device=device)
    print("pos_weight (first 5):", pos_weight[:5].detach().cpu().numpy())

    '''
    # focal loss with logits + pos_weight instead of BCEWithLogitsLoss since focal loss often helps F1 by prioritizing rare classes
     criterion = FocalLossWithLogits(
        gamma=1.5,                   # drop gamma to 1.5 if loss spikes
        alpha=None,                  # keep None first; add later if needed
        pos_weight=pos_weight.to(device),
        reduction="mean",
    )   
    '''

    # BCEWithLogitsLoss with pos_weight, trying this over focal loss to see if simpler loss can still get good F1 (and often trains more stably than focal loss)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Optimizer/scheduler/scaler setup + resume logic for best model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    
    start_epoch = 1
    best_f1 = -1.0

    # if resume specified, load checkpoint .pt file
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])

        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1 = ckpt.get("best_micro_f1", ckpt.get("val_micro_f1", -1.0))
        print(f"Resumed from {args.resume} | start_epoch={start_epoch} | best_f1={best_f1:.4f}")
    
    # train loop and save best model based on val micro-F1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
        
    # change batch size to match target batch size
    target_batch_size = 32
    accum_steps = max(1, target_batch_size // args.batch_size)
    print(f"batch size: {target_batch_size}, accum_steps={accum_steps}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_dt = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch, args.epochs, accum_steps=accum_steps
        )
        val_loss, val_micro_f1 = eval_epoch(
            model, val_loader, criterion, device, epoch, args.epochs, threshold=0.5
        )

        lr_now = optimizer.param_groups[0]["lr"]
        iters = len(train_loader)
        print(
            f"Epoch {epoch}/{args.epochs} | lr={lr_now:.2e} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_micro_f1@0.5={val_micro_f1:.4f} | "
            f"time={train_dt/60:.2f} min | iters={iters}"
        )

        scheduler.step(val_micro_f1)
        # check for best val micro-F1 and save checkpoint
        if val_micro_f1 > best_f1:
            best_f1 = val_micro_f1

            ckpt_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "label_cols": label_cols,
                "val_loss": val_loss,
                "val_micro_f1": val_micro_f1,
                "best_micro_f1": best_f1,
            }

            ckpt_path = out_dir / "best.pt"
            torch.save(ckpt_dict, ckpt_path)
            print("  saved:", ckpt_path)

            # update best-ever across sessions (only when this run improves)
            best_ever_path = Path(args.best_ever)
            prev_best = -1.0

            if best_ever_path.exists():
                prev = torch.load(best_ever_path, map_location="cpu")
                prev_best = prev.get("best_micro_f1", prev.get("val_micro_f1", -1.0))

            if val_micro_f1 > prev_best:
                torch.save(ckpt_dict, best_ever_path)
                print("  updated best-ever:", best_ever_path)


    # load best checkpoint before final eval
    ckpt_path = Path(args.resume) if args.resume else (out_dir / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nLoaded best checkpoint: epoch={ckpt.get('epoch')} | best_val_micro_f1@0.5={ckpt.get('best_micro_f1')}\n")
    
    # collect logits/targets
    val_logits, val_targets = collect_logits_targets(model, val_loader, device)
    test_logits, test_targets = collect_logits_targets(model, test_loader, device)

    # AUROC + mAP (threshold-free) on VAL and TEST
    val_logits_np = val_logits.detach().cpu().numpy()
    val_targets_np = val_targets.detach().cpu().numpy()
    test_logits_np = test_logits.detach().cpu().numpy()
    test_targets_np = test_targets.detach().cpu().numpy()

    val_metrics = compute_auc_map(val_logits_np, val_targets_np, label_cols)
    test_metrics = compute_auc_map(test_logits_np, test_targets_np, label_cols)


    print(f"VAL  macro_AUROC={val_metrics['macro_auroc']:.4f} | macro_mAP={val_metrics['macro_map']:.4f}")
    print(f"TEST macro_AUROC={test_metrics['macro_auroc']:.4f} | macro_mAP={test_metrics['macro_map']:.4f}")

    # tune per-class thresholds on VAL
    thr_vec, val_micro = tune_thresholds_per_class(val_logits, val_targets, label_cols)
    print(f"\nPer-class threshold tuning -> VAL micro-F1={val_micro:.4f}")

    # apply those thresholds to TEST
    val_probs = 1 / (1 + np.exp(-val_logits_np))
    test_probs = 1 / (1 + np.exp(-test_logits_np))

    test_micro = micro_f1_from_probs(test_probs, test_targets_np, thr_vec)
    print(f"TEST micro-F1 (per-class thresholds) = {test_micro:.4f}")

    # compare to a fixed threshold baseline
    for th in [0.5, 0.35, 0.25]:
        v  = micro_f1_from_probs(val_probs,  val_targets_np, th)
        te = micro_f1_from_probs(test_probs, test_targets_np, th)

        print(f"Fixed thr={th:.2f} | VAL micro-F1={v:.4f} | TEST micro-F1={te:.4f}")

    # tune global threshold on VAL (simple + usually boosts F1)
    val_logits_cpu, val_targets_cpu = collect_logits_targets(model, val_loader, device)
    ths = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_thr_f1 = -1.0
    
    for thr in ths:
        f1 = micro_f1_from_logits_cpu(val_logits_cpu, val_targets_cpu, threshold=float(thr))
        if f1 > best_thr_f1:
            best_thr_f1 = f1
            best_thr = float(thr)
    print(f"Best VAL global threshold: {best_thr:.2f} | val_micro_f1={best_thr_f1:.4f}")
    
    # final test eval using tuned threshold
    test_loss, test_micro_f1 = eval_epoch(model, test_loader, criterion, device, threshold=best_thr)
    print(f"test_loss: {test_loss:.4f} | test_micro_f1@{best_thr:.2f}: {test_micro_f1:.4f}")

if __name__ == "__main__":
    # Windows multiprocessing friendliness
    import multiprocessing as mp
    mp.freeze_support()
    main()

# python train.py --epochs 25 --batch_size 32 --num_workers 6 --lr 1e-4 --weight_decay 1e-4
# python train.py --resume runs/best_ever.pt --epochs 50 --batch_size 16 --num_workers 6 --lr 1e-4 --weight_decay 1e-4
# python train.py --resume runs/best_ever.pt --epochs 0 --batch_size 64 --num_workers 6