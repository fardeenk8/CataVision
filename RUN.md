# How to Run the Cataract Detection Project

Follow these steps in order. Use the virtual environment you created.

---

## 1. Activate the environment

```bash
cd "/Users/fardeenkachawa/Documents/Development/Cataract Detection System/cataract-detection"
source venv/bin/activate
```

On Windows (PowerShell): `.\venv\Scripts\Activate.ps1`

---

## 2. Get the Roboflow OculaCare dataset (naked-eye)

1. In Roboflow, download the **OculaCare** cataract dataset as a **Folder Structure** export.

2. Extract it under:

   ```bash
   data/raw_roboflow/
   ```

   so you have:
   - `data/raw_roboflow/train/<class>/*.jpg`
   - `data/raw_roboflow/valid/<class>/*.jpg`
   - `data/raw_roboflow/test/<class>/*.jpg`

---

## 3. Prepare binary labels and splits

```bash
python data/prepare_roboflow_binary.py --data-dir data/raw_roboflow --splits-path data/splits_roboflow.json
```

You should see: "Wrote N train, M val, K test to data/splits_roboflow.json".

---

## 4. Train the models

**Baseline CNN:**
```bash
python scripts/train_baseline.py
```

**EfficientNetB0:**
```bash
python scripts/train_efficientnet.py
```

Checkpoints are saved as:
- `checkpoints/baseline_best.pt`
- `checkpoints/efficientnet_roboflow_best.pt`

Training can take a while; you can stop early with Ctrl+C (best checkpoint is already saved).

---

## 5. Run evaluation

```bash
python scripts/run_evaluation.py
```

Outputs go to `results/evaluation/`: metrics JSON files and ROC/confusion matrix plots.

---

## 6. Start the web demo

```bash
python api/app.py
```

Open in your browser: **http://127.0.0.1:5001**

Upload an eye photo to get prediction, confidence, and Grad-CAM heatmap.  
The demo uses the EfficientNet checkpoint trained on Roboflow; train first (step 4) for meaningful results.

---

## Quick reference (after dataset is in place)

```bash
source venv/bin/activate
python data/prepare_roboflow_binary.py --data-dir data/raw_roboflow --splits-path data/splits_roboflow.json
python scripts/train_baseline.py
python scripts/train_efficientnet.py
python scripts/run_evaluation.py
python api/app.py
```
