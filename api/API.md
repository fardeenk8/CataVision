# Cataract Detection API (for Flutter / mobile)

Naked-eye model only. Run the server with **USE_ROBOFLOW=1** so all requests use the OculaCare (naked-eye) checkpoint.

---

## Base URL

- **Local:** `http://<your-machine-ip>:5001` (e.g. `http://192.168.1.10:5001` for same Wi‑Fi)
- **Production:** `https://your-domain.com` (use HTTPS)

---

## Endpoints

### GET /health

Health check. Use from Flutter to verify the server is up before uploading.

**Response (200):**
```json
{ "status": "ok", "model": "naked-eye" }
```

**Response (503):** Model failed to load.

---

### POST /predict

Upload an eye image; get prediction, confidence, and Grad-CAM overlay.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** one of:
  - `image` (file)
  - `file` (file)

**Optional header (if API_KEY env is set):**
- `X-API-Key: <your-api-key>`

**Response (200):**
```json
{
  "label": "cataract" | "non-cataract",
  "confidence": 0.92,
  "prob_cataract": 0.92,
  "gradcam_b64": "<base64-encoded-PNG>"
}
```

**Errors:**
- **400** – No image in request
- **401** – Invalid or missing API key (when API_KEY is set)
- **500** – Server error (e.g. invalid image, model error)

---

## Production recommendations

1. **HTTPS** – Serve behind nginx/Caddy or a cloud load balancer with TLS. Flutter can use `https://` in the base URL.
2. **Auth** – Set `API_KEY` env on the server; Flutter sends `X-API-Key: <key>` on every `/predict` (and optionally `/health`) request.
3. **Rate limiting** – Add per-IP or per-key limits (e.g. flask-limiter) to avoid abuse.
4. **CORS** – Already enabled for `/predict` and `/health`; restrict origins in production if needed.

---

## Running for mobile testing

```bash
cd cataract-detection
source venv/bin/activate
USE_ROBOFLOW=1 python api/app.py
```

Then in Flutter, set base URL to `http://<your-machine-ip>:5001` (same Wi‑Fi as the phone/emulator).
