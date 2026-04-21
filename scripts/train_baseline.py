"""
Train baseline CNN on the Roboflow OculaCare (naked-eye) dataset.
Calls models.train with BaselineCNN and class weights.
"""
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SPLITS_PATH, BASELINE_CKPT, BATCH_SIZE, LR, EPOCHS, PATIENCE, SEED
from models.baseline_cnn import BaselineCNN
from models.train import load_splits, get_class_weights, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH, help="Path to Roboflow splits JSON")
    parser.add_argument("--checkpoint", type=Path, default=BASELINE_CKPT, help="Output checkpoint path")
    args = parser.parse_args()
    splits = load_splits(args.splits)
    train_samples = splits["train"]
    val_samples = splits["val"]
    class_weights = get_class_weights(train_samples)
    model = BaselineCNN(in_channels=3, channels=(32, 64, 128, 256), num_classes=1)
    run_training(
        model,
        train_samples=train_samples,
        val_samples=val_samples,
        checkpoint_path=args.checkpoint,
        class_weights=class_weights,
        epochs=EPOCHS,
        patience=PATIENCE,
        lr=LR,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    print("Best checkpoint saved to", args.checkpoint)


if __name__ == "__main__":
    main()
