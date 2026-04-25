from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import gc

app = Flask(__name__)
CORS(app)

_model = None
_load_error = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Load model lazily ─────────────────────────────────────────
def load_model():
    global _model, _load_error

    if _model is not None:
        return True

    try:
        model_path = os.path.join(BASE_DIR, "car_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError("car_model.pkl not found")

        print("⏳ Loading model...")
        _model = joblib.load(model_path)
        gc.collect()
        print("✅ Model loaded")

        _load_error = None
        return True

    except Exception as e:
        _load_error = str(e)
        print(f"❌ Model load failed: {e}")
        return False


def get_model():
    if not load_model():
        return None
    return _model


# ── Health check ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    load_model()
    return jsonify({
        "status": "ok" if _model else "error",
        "model_loaded": _model is not None,
        "load_error": _load_error
    })


# ── Predict Price ────────────────────────────────────────────
@app.route("/predict_price", methods=["POST"])
def predict_price():
    model_obj = get_model()
    if model_obj is None:
        return jsonify({"success": False, "error": _load_error or "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid or empty JSON body"}), 400

    # Required fields
    missing = [f for f in ["region", "manufacturer", "fuel"] if not data.get(f)]
    if missing:
        return jsonify({"success": False, "error": f"Missing: {', '.join(missing)}"}), 400

    fuel = data.get("fuel", "").lower()
    is_electric = fuel == "electric"

    input_dict = {
        "region":       data.get("region"),
        "manufacturer": data.get("manufacturer"),
        "model":        data.get("model"),
        "fuel":         fuel,
        "engine_cc":    0 if is_electric else _safe_num(data.get("engine_cc")),
        "max_power":    _safe_num(data.get("max_power")),
        "cylinders":    0 if is_electric else _safe_num(data.get("cylinders")),
        "transmission": data.get("transmission"),
        "type":         data.get("body_type"),
        "drive":        data.get("drive_train"),
        "seats":        _safe_num(data.get("seats")),
        "odometer":     _safe_num(data.get("km_driven")),
        "age":          _safe_num(data.get("age")),
    }

    try:
        prediction = float(model_obj.predict(pd.DataFrame([input_dict]))[0])
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500

    # Confidence range
    mae   = 94338
    lower = max(0, prediction - 0.75 * mae)
    upper = prediction + 0.75 * mae

    km_driven  = _safe_num(data.get("km_driven", 0))
    age        = _safe_num(data.get("age", 0))
    base_price = prediction + (km_driven * 0.5) + (age * 20000)

    breakdown = [b for b in [
        {"l": "Base vehicle value",   "v": round(base_price),       "cls": "neu"},
        {"l": "Mileage adjustment",   "v": round(-km_driven * 0.5), "cls": "neg" if km_driven > 0 else "neu"},
        {"l": "Age depreciation",     "v": round(-age * 20000),     "cls": "neg" if age > 0 else "neu"},
        {"l": "Transmission premium", "v": 15000 if data.get("transmission","").lower() == "automatic" else 0, "cls": "pos"},
        {"l": "Fuel type adjustment", "v": 20000 if is_electric else (10000 if fuel == "hybrid" else 0), "cls": "pos"},
    ] if b["v"] != 0]

    return jsonify({
        "success": True,
        "result": {
            "price":      round(prediction),
            "low":        round(lower),
            "high":       round(upper),
            "confidence": 75,
            "breakdown":  breakdown,
        },
    })


# ── Prediction history (dummy) ───────────────────────────────
@app.route("/predictions/history", methods=["GET"])
def predictions_history():
    return jsonify({"success": True, "count": 0, "predictions": []})


# ── Helper ──────────────────────────────────────────────────
def _safe_num(val, default=0):
    try:
        return float(val) if val not in (None, "", "null") else default
    except (TypeError, ValueError):
        return default


# ── Run locally ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)