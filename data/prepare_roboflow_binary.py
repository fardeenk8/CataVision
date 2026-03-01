"""
Build binary splits from Roboflow Folder Structure export (OculaCare / naked-eye).
Expects: <data_dir>/train/<class>/*.jpg, valid/, test/.
Maps class names: any containing 'cataract' -> 1, else -> 0.
Writes data/splits_roboflow.json for use with the training/evaluation pipeline.
"""
import argparse
import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_ROBOFLOW_DIR, SPLITS_ROBOFLOW_PATH


def collect_from_split(root: Path, split_name: str) -> list:
    """root = e.g. .../train. Returns list of {image_path, label}."""
    samples = []
    if not root.exists():
        return samples
    # Class subfolders (e.g. Cataract, Normal)
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        label = 1 if "cataract" in class_name.lower() else 0
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for f in class_dir.glob(ext):
                samples.append({"image_path": str(f.resolve()), "label": label})
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare Roboflow OculaCare (folder structure) for binary cataract split")
    parser.add_argument("--data-dir", type=Path, default=RAW_ROBOFLOW_DIR,
                        help="Directory containing train/, valid/, test/ (e.g. raw_roboflow/OculaCare.v1i.folder)")
    parser.add_argument("--splits-path", type=Path, default=SPLITS_ROBOFLOW_PATH,
                        help="Output JSON path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = collect_from_split(data_dir / "train", "train")
    val = collect_from_split(data_dir / "valid", "valid")
    test = collect_from_split(data_dir / "test", "test")

    if not train:
        print("No training samples found. Ensure --data-dir contains train/<class>/*.jpg (e.g. train/Cataract, train/Normal)")
        return 1

    splits = {"train": train, "val": val, "test": test}
    args.splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {len(train)} train, {len(val)} val, {len(test)} test to {args.splits_path}")
    return 0


if __name__ == "__main__":
    exit(main())
