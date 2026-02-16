# Cataract Detection System

Naked-eye cataract detection using the Roboflow OculaCare dataset: binary classification (cataract vs non-cataract) from eye photos, with a baseline CNN and EfficientNetB0 transfer learning, evaluation metrics, Grad-CAM explainability, a Flask web API, and a Flutter mobile app.

## Requirements

- Python 3.8+
- CUDA optional (for GPU training)

## Environment

```bash
cd cataract-detection
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For reproducibility, set a fixed seed (optional):

```bash
export PYTHONHASHSEED=42
```

## Dataset: Roboflow OculaCare (naked-eye)

1. In Roboflow, download the **OculaCare** cataract dataset as a **Folder Structure** export.
2. Place the extracted folder under:

   ```bash
   data/raw_roboflow/
   ```

   so you have:

   - `data/raw_roboflow/train/<class>/*.jpg`
   - `data/raw_roboflow/valid/<class>/*.jpg`
   - `data/raw_roboflow/test/<class>/*.jpg`

3. Prepare binary labels and train/val/test split JSON:

```bash
python data/prepare_roboflow_binary.py --data-dir data/raw_roboflow --splits-path data/splits_roboflow.json
```

Output: `data/splits_roboflow.json` with `train`, `val`, and `test` lists of `{ "image_path", "label" }`.

## Training

Train baseline CNN:

```bash
python scripts/train_baseline.py
```

Train EfficientNetB0 (freeze backbone then fine-tune):

```bash
python scripts/train_efficientnet.py
```

Checkpoints are saved under `checkpoints/`:
- `checkpoints/baseline_best.pt`
- `checkpoints/efficientnet_roboflow_best.pt`

Training uses stratified splits, augmentation (in the dataset), class imbalance (BCEWithLogitsLoss with class weights), early stopping (validation ROC-AUC), and checkpointing of the best model.

## Evaluation

Run evaluation for both models on the test set (accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve):

```bash
python scripts/run_evaluation.py
```

Outputs are written to `results/evaluation/`:
- `baseline_metrics.json`, `efficientnet_metrics.json`
- `baseline_roc.png`, `efficientnet_roc.png`
- `baseline_confusion_matrix.png`, `efficientnet_confusion_matrix.png`
- `all_metrics.json` (combined comparison)

## Explainability: Grad-CAM

Grad-CAM for EfficientNetB0 is implemented in `explainability/grad_cam.py` and is used:
- In the Flask demo for each uploaded image (overlay returned as base64).
- You can call `run_gradcam(model, x, device)` from evaluation or other scripts to get the overlay image for any input tensor.

## Demo: Web API + Flutter app

Start the Flask API:

```bash
python api/app.py
```

Then open the web UI at **http://127.0.0.1:5001** or use the Flutter app in `cataract_detection_app/` (see its README / `lib/config.dart` for base URL settings).

- Upload an eye photo.
- The app returns prediction (cataract / non-cataract), confidence, and a Grad-CAM heatmap overlay (displayed in the page).

The demo uses the EfficientNetB0 checkpoint trained on Roboflow; ensure `checkpoints/efficientnet_roboflow_best.pt` exists (run training first).

## Project layout

```
cataract-detection/
├── requirements.txt
├── README.md
├── config.py
├── data/
│   ├── prepare_roboflow_binary.py
│   └── dataset.py
├── models/
│   ├── baseline_cnn.py
│   ├── efficientnet.py
│   └── train.py
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── plots.py
├── explainability/
│   └── grad_cam.py
├── api/
│   ├── app.py
│   ├── templates/
│   └── static/
├── scripts/
│   ├── train_baseline.py
│   ├── train_efficientnet.py
│   └── run_evaluation.py
├── checkpoints/
└── results/
```

## Reproducibility

- Use the same `--seed` in `data/prepare_binary.py` and the same `SEED` in `config.py` (default 42).
- Training scripts use this seed for PyTorch and NumPy.
- Exact commands above reproduce data split, training, evaluation, and demo.
