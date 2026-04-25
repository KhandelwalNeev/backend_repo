# run_locally.py
import joblib, pandas as pd, gc, os

BASE = "."  # folder where your pkl files are

# Shrink dataframe
print("Loading cars_dataframe.pkl...")
df = joblib.load(os.path.join(BASE, "cars_dataframe.pkl"))
print(f"Original size: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB, rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

keep = ["region", "manufacturer", "model", "fuel",
        "engine_cc", "cylinders", "max_power", "transmission",
        "type", "drive", "seats"]

existing = [c for c in keep if c in df.columns]
missing  = [c for c in keep if c not in df.columns]
print(f"Missing columns (will skip): {missing}")

df = df[existing].copy()
for col in df.select_dtypes("object").columns:
    df[col] = df[col].astype("category")
for col in df.select_dtypes("float64").columns:
    df[col] = df[col].astype("float32")
for col in df.select_dtypes("int64").columns:
    df[col] = df[col].astype("int32")

gc.collect()
print(f"Shrunk size: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
joblib.dump(df, os.path.join(BASE, "cars_dataframe.pkl"), compress=("zlib", 6))
print("✅ Saved compressed cars_dataframe.pkl")

# Check model size
model_size = os.path.getsize(os.path.join(BASE, "car_model.pkl")) / 1e6
print(f"\ncar_model.pkl size on disk: {model_size:.1f} MB")
if model_size > 200:
    print("⚠️  Model is very large — consider compressing it too")
    model = joblib.load(os.path.join(BASE, "car_model.pkl"))
    joblib.dump(model, os.path.join(BASE, "car_model.pkl"), compress=("zlib", 3))
    print(f"✅ Compressed model size: {os.path.getsize(os.path.join(BASE, 'car_model.pkl')) / 1e6:.1f} MB")