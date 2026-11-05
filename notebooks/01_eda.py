import pandas as pd

visits = pd.read_csv("data/raw/air_visit_data.csv", parse_dates=["visit_date"])
print(visits.head())
summary = visits.groupby("air_store_id").agg(
    n_rows=("visit_date", "count"),
    first_day=("visit_date", "min"),
    last_day=("visit_date", "max"),
)

print(summary.head())
print(len(summary))
print(summary["n_rows"].describe())
print((summary["n_rows"] < 100).sum())
print((summary["first_day"] > "2016-01-01").sum())
summary["span_days"] = (summary["last_day"] - summary["first_day"]).dt.days + 1
summary["missing_days"] = summary["span_days"] - summary["n_rows"]
summary["missing_share"] = summary["missing_days"] / summary["span_days"]
print(summary["missing_share"].describe())
#one_id = "air_ba937bf13d40fb24"
#one = visits[visits["air_store_id"] == one_id]
#print(summary.loc[one_id])
#print(len(one), one["visit_date"].min(), one["visit_date"].max())
#full = pd.date_range(start=one["visit_date"].min(), end=one["visit_date"].max(), freq="D")
#print(len(full))
#missing = full.difference(one["visit_date"])
#print(len(missing))
#print(missing.day_name().value_counts())
all_missing = []
closed_rows = []
for store_id, group in visits.groupby("air_store_id"):
    full = pd.date_range(start=group["visit_date"].min(), end=group["visit_date"].max(), freq="D")
    missing = full.difference(group["visit_date"])
    all_missing.extend(missing)
    if len(missing) == 0:
        continue
    counts = missing.day_name().value_counts()
    closed_rows.append({
        "air_store_id": store_id,
        "n_missing": len(missing),
        "top_day": counts.index[0],
        "top_share": counts.iloc[0] / counts.sum(),
    })
all_missing = pd.DatetimeIndex(all_missing)
all_missing = pd.DatetimeIndex(all_missing)
print(len(all_missing), summary["missing_days"].sum())
print(all_missing.day_name().value_counts())
closed = pd.DataFrame(closed_rows)
print(len(closed))
print(closed["top_share"].describe())

clear = closed[closed["top_share"] > 0.5]
print(len(clear))
print(clear["top_day"].value_counts())

clear10 = clear[clear["n_missing"] >= 10]
print(len(clear10))
for t in [0.4, 0.5, 0.6]:
    print(t, (closed["top_share"] > t).sum())
per_day = visits.groupby("visit_date")["air_store_id"].nunique()
print(per_day.sort_values().head(10))

print(summary["first_day"].dt.to_period("M").value_counts().sort_index())

print(summary.sort_values("missing_share", ascending=False).head(10))
visits["weekday"] = visits["visit_date"].dt.day_name()
by_day = visits.groupby("weekday")["visitors"].agg(["mean", "median", "count"])
print(by_day)
print(by_day["mean"].max() / by_day["mean"].min())