"""
Flask API for cataract detection (naked-eye model).
POST /predict: upload image, returns prediction, confidence, Grad-CAM (base64).
GET /health: health check for mobile app.
Optional: X-API-Key header for auth (set API_KEY env).
"""
import sys
print("[1/8] Starting imports...", flush=True)

from pathlib import Path
import base64
import io
import os
print("[2/8] stdlib done...", flush=True)

import torch
print("[3/8] torch done...", flush=True)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
print("[4/8] flask/PIL done...", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, EFFICIENTNET_CKPT
print("[5/8] config done...", flush=True)

from models.efficientnet import EfficientNetCataract
print("[6/8] model done...", flush=True)

from data.dataset import get_val_test_transform
print("[7/8] dataset done...", flush=True)

from explainability.grad_cam import run_gradcam
print("[8/8] grad_cam done...", flush=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
CORS(app, resources={r"/predict": {"origins": "*"}, r"/health": {"origins": "*"}})  # Flutter / any origin

# Load model once at startup
_model = None
_transform = None
_device = None


def get_model():
    global _model, _transform, _device
    if _model is None:
        ckpt_path = EFFICIENTNET_CKPT
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = EfficientNetCataract(num_classes=1, pretrained=False, freeze_backbone=False)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=_device)
            _model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:
            import warnings
            warnings.warn(
                f"Checkpoint not found: {ckpt_path}. "
                "Run training first (scripts/train_efficientnet.py) for meaningful predictions.",
                UserWarning,
            )
        _model = _model.to(_device)
        _model.eval()
        _transform = get_val_test_transform()
    return _model, _transform, _device


def image_to_tensor(image_bytes):
    """Load image from bytes, apply val transform, return (1, 3, H, W)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model, transform, _ = get_model()
    x = transform(img).unsqueeze(0)
    return x


def _check_api_key():
    """Optional API key: set API_KEY env; client sends X-API-Key header."""
    key = os.environ.get("API_KEY", "").strip()
    if not key:
        return True
    return request.headers.get("X-API-Key", "").strip() == key


@app.route("/health", methods=["GET"])
def health():
    """Health check for mobile app. Returns 200 when server and model are ready."""
    try:
        get_model()
        return jsonify({"status": "ok", "model": "naked-eye"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not _check_api_key():
        return jsonify({"error": "Invalid or missing API key"}), 401
    if "image" not in request.files and "file" not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files.get("image") or request.files.get("file")
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    try:
        image_bytes = file.read()
        x = image_to_tensor(image_bytes)
        model, transform, device = get_model()
        x_dev = x.to(device)
        with torch.no_grad():
            logits = model(x_dev)
        prob = torch.sigmoid(logits).item()
        # If labels were inverted in training (e.g. dataset folder swap), set FLIP_CATARACT_PREDICTION=1
        import os
        if os.environ.get("FLIP_CATARACT_PREDICTION", "").strip().lower() in ("1", "true", "yes"):
            prob = 1.0 - prob
        # Ensure we use naked-eye model when called from mobile (or run server with USE_ROBOFLOW=1)
        label = "cataract" if prob >= 0.5 else "non-cataract"
        confidence = float(prob) if label == "cataract" else float(1 - prob)
        # Grad-CAM overlay
        overlay = run_gradcam(model, x_dev, device=device)
        from PIL import Image as PILImage
        pil_overlay = PILImage.fromarray(overlay)
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return jsonify({
            "label": label,
            "confidence": round(confidence, 4),
            "prob_cataract": round(prob, 4),
            "gradcam_b64": gradcam_b64,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Cataract Detection Flask API starting app.run()...", flush=True)
    app.run(host="0.0.0.0", port=5001, debug=False)
