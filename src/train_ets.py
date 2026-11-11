import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.metrics import compute_rmsle, compute_mae, compute_rmse

# 1. Load data
base_dir = Path(__file__).resolve().parents[1]
data_path = base_dir / "data" / "raw" / "air_visit_data.csv"
visit = pd.read_csv(data_path, parse_dates=["visit_date"])

# 2. Filter to Top 10 most active restaurants to benchmark ETS
top_stores = visit["air_store_id"].value_counts().head(10).index.tolist()

cutoff = pd.Timestamp("2017-03-14")
val_end = pd.Timestamp("2017-04-22")

# Build a continuous daily date range for the 10 stores
full_dates = pd.date_range("2016-01-01", val_end, freq="D")

ets_actuals = []
ets_preds = []
lgbm_baseline_preds = []

print("Fitting ETS models across top benchmark restaurants...")

for store_id in top_stores:
    store_df = visit[visit["air_store_id"] == store_id].set_index("visit_date")
    # Reindex to regular daily frequency; fill unobserved days with forward-fill/interpolation for ETS
    ts = store_df["visitors"].reindex(full_dates)
    train_ts = ts.loc[:cutoff].ffill().bfill()
    val_actual = ts.loc[cutoff + pd.Timedelta(days=1):val_end]
    
    # Fit ETS with additive weekly seasonality (season_periods=7)
    try:
        model = ExponentialSmoothing(
            train_ts,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated"
        ).fit()
        forecast = model.forecast(steps=len(val_actual))
    except Exception:
        # Fallback to simple exponential smoothing if seasonality fails
        model = ExponentialSmoothing(train_ts, trend=None, seasonal=None).fit()
        forecast = model.forecast(steps=len(val_actual))
    
    # Only score on observed days (ignore days where the restaurant was closed/unrecorded)
    obs_mask = val_actual.notna()
    if obs_mask.sum() > 0:
        ets_actuals.extend(val_actual[obs_mask].values)
        ets_preds.extend(forecast[obs_mask].values)

ets_actuals = np.array(ets_actuals)
ets_preds = np.clip(np.array(ets_preds), 0, None)

print("\n" + "="*50)
print("     ETS BENCHMARK RESULTS (Top 10 Stores)")
print("="*50)
print(f"Sample size: {len(ets_actuals)} observations")
print(f"RMSLE: {compute_rmsle(ets_actuals, ets_preds):.4f}")
print(f"MAE:   {compute_mae(ets_actuals, ets_preds):.4f}")
print(f"RMSE:  {compute_rmse(ets_actuals, ets_preds):.4f}")
print("="*50)