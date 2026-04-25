from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import threading
import gc

app = Flask(__name__)
CORS(app)

_model = None
_cars_df = None
_load_lock = threading.Lock()
_load_error = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_resources():
    global _model, _cars_df, _load_error

    if _model is not None and _cars_df is not None:
        return True

    with _load_lock:
        if _model is not None and _cars_df is not None:
            return True
        try:
            # ── Step 1: Check files exist ──────────────────────────────────
            model_path = os.path.join(BASE_DIR, "car_model.pkl")
            df_path    = os.path.join(BASE_DIR, "cars_dataframe.pkl")

            print("📂 BASE_DIR:", BASE_DIR)
            print("📂 Files:", os.listdir(BASE_DIR))

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"car_model.pkl not found at {model_path}")
            if not os.path.exists(df_path):
                raise FileNotFoundError(f"cars_dataframe.pkl not found at {df_path}")

            print(f"📦 model size : {os.path.getsize(model_path)  / 1e6:.1f} MB")
            print(f"📦 df size    : {os.path.getsize(df_path) / 1e6:.1f} MB")

            # ── Step 2: Load model ─────────────────────────────────────────
            print("⏳ Loading car_model.pkl ...")
            _model = joblib.load(model_path)
            gc.collect()
            print("✅ Model loaded")

            # ── Step 3: Load dataframe ─────────────────────────────────────
            print("⏳ Loading cars_dataframe.pkl ...")
            df = joblib.load(df_path)
            print(f"   Raw df: {len(df)} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
            print(f"   Columns: {list(df.columns)}")

            # ── Step 4: Keep only needed columns ───────────────────────────
            keep_cols = [
                "region", "manufacturer", "model", "fuel",
                "engine_cc", "cylinders", "max_power", "transmission",
                "type", "drive", "seats",
            ]
            existing = [c for c in keep_cols if c in df.columns]
            missing_cols = [c for c in keep_cols if c not in df.columns]
            if missing_cols:
                print(f"⚠️  Columns not found (will be skipped): {missing_cols}")

            _cars_df = df[existing].copy()
            del df
            gc.collect()

            # ── Step 5: Shrink dtypes ──────────────────────────────────────
            for col in _cars_df.select_dtypes(include=["object"]).columns:
                _cars_df[col] = _cars_df[col].astype("category")
            for col in _cars_df.select_dtypes(include=["float64"]).columns:
                _cars_df[col] = _cars_df[col].astype("float32")
            for col in _cars_df.select_dtypes(include=["int64"]).columns:
                _cars_df[col] = _cars_df[col].astype("int32")

            gc.collect()
            print(f"✅ DataFrame ready — {len(_cars_df)} rows, {_cars_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
            _load_error = None
            return True

        except Exception as e:
            _load_error = str(e)
            print(f"❌ Load failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def get_resources():
    if not load_resources():
        return None, None
    return _model, _cars_df


# ── Health check — also triggers load so you can see errors immediately ───────
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    load_resources()  # ← trigger load so health shows real status
    files_present = {
        "car_model.pkl":      os.path.exists(os.path.join(BASE_DIR, "car_model.pkl")),
        "cars_dataframe.pkl": os.path.exists(os.path.join(BASE_DIR, "cars_dataframe.pkl")),
    }
    return jsonify({
        "status":        "ok" if _model is not None else "error",
        "model_loaded":  _model is not None,
        "rows":          len(_cars_df) if _cars_df is not None else 0,
        "load_error":    _load_error,
        "files_present": files_present,
        "base_dir":      BASE_DIR,
        "all_files":     os.listdir(BASE_DIR),
    })


# ── /predict/options ──────────────────────────────────────────────────────────
@app.route("/predict/options", methods=["GET"])
def predict_options():
    _, df = get_resources()
    if df is None:
        return jsonify({"success": False, "error": _load_error or "Model not loaded"}), 503

    return jsonify({
        "success":       True,
        "regions":       sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else [],
        "manufacturers": sorted(df["manufacturer"].dropna().unique().tolist()) if "manufacturer" in df.columns else [],
    })


# ── /predict/models ───────────────────────────────────────────────────────────
@app.route("/predict/models", methods=["GET"])
def predict_models():
    _, df = get_resources()
    if df is None:
        return jsonify({"success": False, "error": _load_error or "Model not loaded"}), 503

    manufacturer = request.args.get("manufacturer", "").strip()
    model_name   = request.args.get("model", "").strip()

    filtered = df
    if manufacturer and "manufacturer" in df.columns:
        filtered = df[df["manufacturer"] == manufacturer]

    models = sorted(filtered["model"].dropna().unique().tolist()) if "model" in filtered.columns else []

    options = {}
    if manufacturer and model_name and "model" in filtered.columns:
        mdf = filtered[filtered["model"] == model_name]

        def uniq(col):
            if col not in mdf.columns:
                return []
            try:
                vals = mdf[col].dropna().unique().tolist()
                vals = [v.item() if hasattr(v, "item") else v for v in vals]
                return sorted(vals)
            except TypeError:
                return sorted(str(v) for v in mdf[col].dropna().unique().tolist())

        options = {
            "fuels":         uniq("fuel"),
            "engine_ccs":    uniq("engine_cc"),
            "cylinders":     uniq("cylinders"),
            "max_powers":    uniq("max_power"),
            "transmissions": uniq("transmission"),
            "body_types":    uniq("type"),
            "drivetrains":   uniq("drive"),
            "seats":         uniq("seats"),
        }

    return jsonify({"success": True, "models": models, "options": options})


# ── /predict_price ────────────────────────────────────────────────────────────
@app.route("/predict_price", methods=["POST"])
def predict_price():
    model_obj, df = get_resources()
    if model_obj is None or df is None:
        return jsonify({"success": False, "error": _load_error or "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid or empty JSON body"}), 400

    missing = [f for f in ["region", "manufacturer", "fuel"] if not data.get(f)]
    if missing:
        return jsonify({"success": False, "error": f"Missing: {', '.join(missing)}"}), 400

    fuel        = data.get("fuel", "").lower()
    is_electric = fuel == "electric"

    check = df
    for col, key in [("manufacturer","manufacturer"),("model","model"),("fuel","fuel"),("transmission","transmission")]:
        val = data.get(key)
        if val and col in check.columns:
            check = check[check[col] == val]
    if len(check) == 0:
        return jsonify({"success": False, "error": "No training data for this combination."}), 400

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


# ── /predictions/history ──────────────────────────────────────────────────────
@app.route("/predictions/history", methods=["GET"])
def predictions_history():
    return jsonify({"success": True, "count": 0, "predictions": []})


# ── helper ────────────────────────────────────────────────────────────────────
def _safe_num(val, default=0):
    try:
        return float(val) if val not in (None, "", "null") else default
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)