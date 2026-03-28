"""
Load test set and checkpoint, run inference, collect logits/labels.
Returns y_true, y_prob for metrics and plots.
"""
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BATCH_SIZE
from data.dataset import CataractDataset, get_val_test_transform
from .metrics import compute_metrics
from .plots import plot_roc_curve, plot_confusion_matrix


def load_model_and_predict(model, checkpoint_path: Path, test_samples, device=None):
    """Run model on test set; return y_true, y_prob."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    ds = CataractDataset(test_samples, transform=get_val_test_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().squeeze()
            all_probs.append(probs)
            all_labels.append(y)
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    return y_true, y_prob


def evaluate_and_save(model, model_name: str, checkpoint_path: Path, test_samples, output_dir: Path):
    """Load model, run on test set, compute metrics, save plots and JSON."""
    y_true, y_prob = load_model_and_predict(model, checkpoint_path, test_samples)
    metrics = compute_metrics(y_true, y_prob)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_roc_curve(y_true, y_prob, output_dir / f"{model_name}_roc.png", title=f"{model_name} ROC")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        output_dir / f"{model_name}_confusion_matrix.png",
        title=f"{model_name} Confusion Matrix",
    )
    return metrics
