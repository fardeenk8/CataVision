# Cataract Detection (Flutter)

Mobile app for naked-eye cataract screening. Upload a photo of the eye; get prediction and Grad-CAM overlay from the Flask API.

## Prerequisites

- Flutter SDK (3.0+)
- Flask API running with **naked-eye model** (`USE_ROBOFLOW=1 python api/app.py`)

## Setup

1. **Set API base URL** in `lib/config.dart`:
   - **Physical device (same Wi‑Fi):** `http://<your-machine-ip>:5001` (e.g. `http://192.168.1.10:5001`)
   - **Android emulator:** `http://10.0.2.2:5001`
   - **iOS simulator:** `http://localhost:5001`

2. **Optional:** If your server uses API key auth, set `kApiKey` in `lib/config.dart`.

## Run

```bash
cd cataract_detection_app
flutter pub get
flutter run
```

If the Android build fails (e.g. missing styles or launcher icon), regenerate platform files:

```bash
flutter create . --project-name cataract_detection_app
```

Then ensure `android/app/src/main/AndroidManifest.xml` includes `INTERNET`, `CAMERA`, and `READ_MEDIA_IMAGES` permissions.

## Flow

1. **Home** – Health check; then "Take photo" or "Choose from gallery".
2. **Result** – Prediction (cataract / no cataract), confidence %, and Grad-CAM overlay.
3. Disclaimer: not a medical diagnosis.

## API

- `GET /health` – Server health (used on app start).
- `POST /predict` – Multipart `image`; returns `label`, `confidence`, `prob_cataract`, `gradcam_b64`.

See `api/API.md` in the parent project for full API details.
