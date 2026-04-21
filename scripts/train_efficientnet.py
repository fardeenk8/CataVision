"""
Train EfficientNetB0 transfer learning on Roboflow OculaCare (naked-eye images).
Freeze backbone for FREEZE_EPOCHS then fine-tune.
Use --phase2-only to skip phase 1 and continue fine-tuning from an existing checkpoint.
"""
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config import SPLITS_PATH, EFFICIENTNET_CKPT, BATCH_SIZE, LR, EPOCHS, PATIENCE, SEED, FREEZE_EPOCHS
from models.efficientnet import EfficientNetCataract
from models.train import load_splits, get_class_weights, run_training


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 for cataract classification")
    parser.add_argument("--phase2-only", action="store_true", help="Skip phase 1; load checkpoint and run phase 2 (fine-tune) only")
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH, help="Path to Roboflow splits JSON")
    parser.add_argument("--checkpoint", type=Path, default=EFFICIENTNET_CKPT, help="Output checkpoint path (Roboflow EfficientNet)")
    args = parser.parse_args()

    splits = load_splits(args.splits)
    train_samples = splits["train"]
    val_samples = splits["val"]
    class_weights = get_class_weights(train_samples)

    if not args.phase2_only:
        # Phase 1: freeze backbone
        model = EfficientNetCataract(num_classes=1, pretrained=True, freeze_backbone=True)
        run_training(
            model,
            train_samples=train_samples,
            val_samples=val_samples,
        checkpoint_path=args.checkpoint,
        class_weights=class_weights,
        epochs=FREEZE_EPOCHS,
            patience=PATIENCE,
            lr=LR,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )

    # Phase 2: unfreeze and fine-tune (load best and continue)
    if not args.checkpoint.exists():
        print("No checkpoint found. Run without --phase2-only to train phase 1 first.")
        return
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = EfficientNetCataract(num_classes=1, pretrained=True, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    run_training(
        model,
        train_samples=train_samples,
        val_samples=val_samples,
        checkpoint_path=args.checkpoint,
        class_weights=class_weights,
        epochs=EPOCHS,
        patience=PATIENCE,
        lr=LR * 0.1,  # lower LR for fine-tuning
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    print("Best checkpoint saved to", args.checkpoint)


if __name__ == "__main__":
    main()
