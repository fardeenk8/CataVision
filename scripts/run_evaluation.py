"""
Evaluate baseline and EfficientNet (both trained on Roboflow OculaCare) on the test set.
Saves metrics and plots to results/evaluation/.
"""
from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SPLITS_PATH, BASELINE_CKPT, EFFICIENTNET_CKPT, EVAL_OUTPUT_DIR
from models.baseline_cnn import BaselineCNN
from models.efficientnet import EfficientNetCataract
from evaluation.evaluate import evaluate_and_save


def load_splits(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH, help="Splits JSON (Roboflow OculaCare)")
    parser.add_argument("--checkpoint-baseline", type=Path, default=BASELINE_CKPT)
    parser.add_argument("--checkpoint-efficientnet", type=Path, default=EFFICIENTNET_CKPT)
    parser.add_argument("--output-dir", type=Path, default=EVAL_OUTPUT_DIR, help="Where to save metrics and plots")
    args = parser.parse_args()

    splits = load_splits(args.splits)
    test_samples = splits["test"]
    results = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_baseline.exists():
        model = BaselineCNN(in_channels=3, channels=(32, 64, 128, 256), num_classes=1)
        metrics = evaluate_and_save(model, "baseline", args.checkpoint_baseline, test_samples, args.output_dir)
        results["baseline"] = metrics
        print("Baseline:", metrics)
    else:
        print("Baseline checkpoint not found:", args.checkpoint_baseline)

    if args.checkpoint_efficientnet.exists():
        model = EfficientNetCataract(num_classes=1, pretrained=False, freeze_backbone=False)
        metrics = evaluate_and_save(model, "efficientnet", args.checkpoint_efficientnet, test_samples, args.output_dir)
        results["efficientnet"] = metrics
        print("EfficientNet:", metrics)
    else:
        print("EfficientNet checkpoint not found:", args.checkpoint_efficientnet)

    with open(args.output_dir / "all_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Outputs saved to", args.output_dir)


if __name__ == "__main__":
    main()
