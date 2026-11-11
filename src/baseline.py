import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from pathlib import Path
from src.metrics import compute_rmsle, compute_mae, compute_rmse

# 1. Locate and load the air_visit data
base_dir = Path(__file__).resolve().parents[1]
data_path = base_dir / "data" / "raw" / "air_visit_data.csv"
print(f"Loading data from: {data_path}")

visit = pd.read_csv(data_path, parse_dates=["visit_date"])

# 2. Strict chronological split (39-day validation window matching test horizon)
cutoff = pd.Timestamp("2017-03-14")
val_end = pd.Timestamp("2017-04-22")

train = visit[visit["visit_date"] <= cutoff].copy()
valid = visit[(visit["visit_date"] > cutoff) & (visit["visit_date"] <= val_end)].copy()

print(f"Train samples: {len(train):,} (up to {cutoff.date()})")
print(f"Validation samples: {len(valid):,} ({valid['visit_date'].min().date()} to {valid['visit_date'].max().date()})")

# -------------------------------------------------------------------------
# Baseline 1: Store Global Mean
# -------------------------------------------------------------------------
store_means = train.groupby("air_store_id")["visitors"].mean()
global_mean = train["visitors"].mean()

valid["pred_mean"] = valid["air_store_id"].map(store_means).fillna(global_mean)

# -------------------------------------------------------------------------
# Baseline 2: Same-Weekday Historical Median per Store
# (Captures strong weekly seasonality observed in EDA)
# -------------------------------------------------------------------------
train["day_of_week"] = train["visit_date"].dt.day_name()
valid["day_of_week"] = valid["visit_date"].dt.day_name()

weekday_median = train.groupby(["air_store_id", "day_of_week"])["visitors"].median().reset_index()
weekday_median.rename(columns={"visitors": "pred_weekday_median"}, inplace=True)

store_medians = train.groupby("air_store_id")["visitors"].median()
global_median = train["visitors"].median()

valid = valid.merge(weekday_median, on=["air_store_id", "day_of_week"], how="left")
valid["pred_weekday_median"] = valid["pred_weekday_median"].fillna(
    valid["air_store_id"].map(store_medians)
).fillna(global_median)

# -------------------------------------------------------------------------
# Baseline 3: Recent 14-Day Store Mean (Short-term moving average)
# -------------------------------------------------------------------------
recent_cutoff = cutoff - pd.Timedelta(days=14)
recent_train = train[train["visit_date"] > recent_cutoff]
recent_means = recent_train.groupby("air_store_id")["visitors"].mean()

valid["pred_recent_14d"] = valid["air_store_id"].map(recent_means).fillna(
    valid["air_store_id"].map(store_means)
).fillna(global_mean)

# -------------------------------------------------------------------------
# Evaluate all 3 baselines
# -------------------------------------------------------------------------
models = [
    ("Store Overall Mean", "pred_mean"),
    ("Same-Weekday Median", "pred_weekday_median"),
    ("Recent 14-Day Mean", "pred_recent_14d"),
]

results = []
y_true = valid["visitors"].values

for name, col in models:
    y_pred = valid[col].values
    results.append({
        "Model": name,
        "RMSLE": round(compute_rmsle(y_true, y_pred), 4),
        "MAE": round(compute_mae(y_true, y_pred), 4),
        "RMSE": round(compute_rmse(y_true, y_pred), 4),
    })

results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("      VALIDATION BASELINE RESULTS (39-DAY HORIZON)")
print("="*50)
print(results_df.to_string(index=False))
print("="*50)