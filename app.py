from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import onnxruntime as rt
import os

app = Flask(__name__)
CORS(app)

_session = None
_input_name = None
_load_error = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Load ONNX model lazily ───────────────────────────────────
def load_model():
    global _session, _input_name, _load_error

    if _session is not None:
        return True

    try:
        model_path = os.path.join(BASE_DIR, "car_model.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError("car_model.onnx not found")

        print("⏳ Loading ONNX model...")
        _session = rt.InferenceSession(model_path)

        _input_name = _session.get_inputs()[0].name

        print("✅ ONNX model loaded")
        print("INPUT INFO:", _session.get_inputs())

        _load_error = None
        return True

    except Exception as e:
        _load_error = str(e)
        print(f"❌ Model load failed: {e}")
        return False


def get_session():
    if not load_model():
        return None
    return _session


# ── Health check ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _session is not None,
        "load_error": _load_error
    })


# ── Predict Price ────────────────────────────────────────────
@app.route("/predict_price", methods=["POST"])
def predict_price():
    session = get_session()
    if session is None:
        return jsonify({"success": False, "error": _load_error or "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid JSON"}), 400

    try:
        fuel = data.get("fuel", "").lower()
        is_electric = fuel == "electric"

        # ✅ IMPORTANT: Proper ONNX input (column-wise dictionary)
        input_data = {
            "region":       np.array([data.get("region", "")], dtype=object),
            "manufacturer": np.array([data.get("manufacturer", "")], dtype=object),
            "model":        np.array([data.get("model", "")], dtype=object),
            "fuel":         np.array([fuel], dtype=object),
            "engine_cc":    np.array([0 if is_electric else _safe_num(data.get("engine_cc"))], dtype=np.float32),
            "max_power":    np.array([_safe_num(data.get("max_power"))], dtype=np.float32),
            "cylinders":    np.array([0 if is_electric else _safe_num(data.get("cylinders"))], dtype=np.float32),
            "transmission": np.array([data.get("transmission", "")], dtype=object),
            "type":         np.array([data.get("body_type", "")], dtype=object),
            "drive":        np.array([data.get("drive_train", "")], dtype=object),
            "seats":        np.array([_safe_num(data.get("seats"))], dtype=np.float32),
            "odometer":     np.array([_safe_num(data.get("km_driven"))], dtype=np.float32),
            "age":          np.array([_safe_num(data.get("age"))], dtype=np.float32),
        }

        # Run inference
        outputs = session.run(None, input_data)
        prediction = float(outputs[0][0])

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"success": False, "error": str(e)}), 500

    # Confidence range
    mae   = 94338
    lower = max(0, prediction - 0.75 * mae)
    upper = prediction + 0.75 * mae

    km_driven  = _safe_num(data.get("km_driven", 0))
    age        = _safe_num(data.get("age", 0))
    base_price = prediction + (km_driven * 0.5) + (age * 20000)

    breakdown = [b for b in [
        {"l": "Base vehicle value",   "v": round(base_price),       "cls": "neu"},
        {"l": "Mileage adjustment",   "v": round(-km_driven * 0.5), "cls": "neg"},
        {"l": "Age depreciation",     "v": round(-age * 20000),     "cls": "neg"},
    ] if b["v"] != 0]

    return jsonify({
        "success": True,
        "result": {
            "price": round(prediction),
            "low": lower,
            "high": upper,
            "confidence": 75,
            "breakdown": breakdown,
        },
    })


# ── Helper ──────────────────────────────────────────────────
def _safe_num(val, default=0):
    try:
        return float(val) if val not in (None, "", "null") else default
    except:
        return default


# ── Run locally ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
