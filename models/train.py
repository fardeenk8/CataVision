"""
Shared training loop: augmentation (in dataset), class imbalance, early stopping, checkpointing.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SPLITS_PATH, CHECKPOINTS_DIR, IMAGE_SIZE, BATCH_SIZE, LR, EPOCHS, PATIENCE, SEED,
    ensure_dirs,
)
from data.dataset import CataractDataset, get_train_transform, get_val_test_transform


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_splits(splits_path: Path):
    with open(splits_path) as f:
        return json.load(f)


def get_class_weights(samples):
    """Compute class weights for BCE (inverse frequency)."""
    labels = [s["label"] for s in samples]
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = len(labels) - n_pos
    n = len(labels)
    if n_pos == 0:
        return torch.tensor([1.0, 1.0])
    w_pos = n / (2 * n_pos)
    w_neg = n / (2 * n_neg)
    return torch.tensor([w_neg, w_pos])  # index 0 = neg, 1 = pos


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits).squeeze()
    labels = torch.cat(all_labels).squeeze()
    probs = torch.sigmoid(logits)
    # ROC-AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels.numpy(), probs.numpy())
    except Exception:
        auc = 0.0
    return total_loss / len(loader), auc


def run_training(
    model,
    train_samples,
    val_samples,
    checkpoint_path: Path,
    device=None,
    class_weights=None,
    epochs=EPOCHS,
    patience=PATIENCE,
    lr=LR,
    batch_size=BATCH_SIZE,
    seed=SEED,
):
    set_seed(seed)
    ensure_dirs()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CataractDataset(train_samples, transform=get_train_transform())
    val_ds = CataractDataset(val_samples, transform=get_val_test_transform())
    pin_memory = device.type == "cuda"  # MPS (Apple) does not support pin_memory
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    if class_weights is not None:
        class_weights = class_weights.to(device)
        pos_weight = (class_weights[1] / class_weights[0]).unsqueeze(0)  # scalar for positive class
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_auc = 0.0
    epochs_no_improve = 0

    print(f"Training on {device} | train batches: {len(train_loader)} | val batches: {len(val_loader)}", flush=True)
    print(f"Starting training for up to {epochs} epochs (early stop patience={patience})...", flush=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} ...", flush=True)
        train_loss = train_epoch(model, loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
        val_loss, val_auc = validate(model, loader=val_loader, criterion=criterion, device=device)
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, checkpoint_path)
        else:
            epochs_no_improve += 1
        print(f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  best_auc={best_auc:.4f}")
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    return best_auc
