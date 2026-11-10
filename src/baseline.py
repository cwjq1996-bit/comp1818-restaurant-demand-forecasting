
from pathlib import Path

import pandas as pd

from src.metrics import compute_rmsle, compute_rmse, compute_mae

# Load the restaurant visitor data
data_path = Path(__file__).resolve().parents[1] / "data/raw/air_visit_data.csv"

visits = pd.read_csv(data_path, parse_dates=["visit_date"])

# Chronological training and validation split
cutoff = pd.Timestamp("2017-03-12")

train = visits[visits["visit_date"] < cutoff].copy()
valid = visits[visits["visit_date"] >= cutoff].copy()

# Calculate each restaurant's average using training data only
store_means = train.groupby("air_store_id")["visitors"].mean()
global_mean = train["visitors"].mean()

# Predict validation visits using historical averages
valid["prediction"] = (
    valid["air_store_id"]
    .map(store_means)
    .fillna(global_mean)
    .clip(lower=0)
)

# Evaluate predictions
print("Training rows:", len(train))
print("Validation rows:", len(valid))

if valid.empty:
    raise ValueError("Validation set is empty. Check the cutoff date.")

print("RMSLE:", compute_rmsle(valid["visitors"], valid["prediction"]))
print("RMSE:", compute_rmse(valid["visitors"], valid["prediction"]))
print("MAE:", compute_mae(valid["visitors"], valid["prediction"]))
