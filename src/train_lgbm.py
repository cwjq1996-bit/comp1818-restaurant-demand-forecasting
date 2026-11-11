import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import lightgbm as lgb
from src.metrics import compute_rmsle, compute_mae, compute_rmse

# 1. Load data
base_dir = Path(__file__).resolve().parents[1]
data_dir = base_dir / "data" / "raw"

print("Loading dataset...")
air_visit = pd.read_csv(data_dir / "air_visit_data.csv", parse_dates=["visit_date"])
date_info = pd.read_csv(data_dir / "date_info.csv", parse_dates=["calendar_date"])
store_info = pd.read_csv(data_dir / "air_store_info.csv")

# 2. Merge calendar information
date_info.rename(columns={"calendar_date": "visit_date"}, inplace=True)
df = air_visit.merge(date_info, on="visit_date", how="left")
df = df.merge(store_info, on="air_store_id", how="left")

# 3. Feature Engineering (Strictly leakage-free)
# Calendar features
df["day_of_week"] = df["visit_date"].dt.dayofweek
df["month"] = df["visit_date"].dt.month
df["day"] = df["visit_date"].dt.day
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Encode categoricals
for col in ["air_store_id", "air_genre_name", "air_area_name"]:
    df[col] = df[col].astype("category")

# Target variable: log1p(visitors) so optimizing RMSE directly optimizes RMSLE!
df["target"] = np.log1p(df["visitors"])

# 4. Chronological train/validation split (same 39-day window as baselines)
cutoff = pd.Timestamp("2017-03-14")
val_end = pd.Timestamp("2017-04-22")

train_mask = df["visit_date"] <= cutoff
val_mask = (df["visit_date"] > cutoff) & (df["visit_date"] <= val_end)

# Calculate store historical statistics strictly from training period
store_stats = df[train_mask].groupby("air_store_id", observed=False)["target"].agg(
    store_target_mean="mean",
    store_target_median="median",
    store_target_std="std"
).reset_index()

df = df.merge(store_stats, on="air_store_id", how="left")

# Features list
feature_cols = [
    "air_store_id",
    "air_genre_name",
    "air_area_name",
    "day_of_week",
    "month",
    "day",
    "is_weekend",
    "holiday_flg",
    "store_target_mean",
    "store_target_median",
    "store_target_std",
]

X_train = df.loc[train_mask, feature_cols]
y_train = df.loc[train_mask, "target"]

X_val = df.loc[val_mask, feature_cols]
y_val = df.loc[val_mask, "target"]
y_val_actual = df.loc[val_mask, "visitors"].values

print(f"Training features shape: {X_train.shape}")
print(f"Validation features shape: {X_val.shape}")

# 5. Train LightGBM model
print("\nTraining LightGBM model...")
model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)

# 6. Predict and transform back from log scale
val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log)
val_preds = np.clip(val_preds, 0, None)

# 7. Evaluate
rmsle = compute_rmsle(y_val_actual, val_preds)
mae = compute_mae(y_val_actual, val_preds)
rmse = compute_rmse(y_val_actual, val_preds)

print("\n" + "="*50)
print("          LIGHTGBM MODEL RESULTS")
print("="*50)
print(f"RMSLE: {rmsle:.4f}")
print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print("="*50)

# Feature Importances
imp = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nTop Features by Importance:")
print(imp.head(6).to_string(index=False))