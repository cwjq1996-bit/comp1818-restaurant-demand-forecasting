import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

# 1. Summary results
results = [
    {"Model": "Store Mean", "Family": "Baseline", "RMSLE": 0.6362},
    {"Model": "14-Day Mean", "Family": "Baseline", "RMSLE": 0.6236},
    {"Model": "Weekday Median", "Family": "Baseline", "RMSLE": 0.5541},
    {"Model": "LightGBM", "Family": "Machine Learning", "RMSLE": 0.5535},
]
df_res = pd.DataFrame(results)

# Create output directory
reports_dir = Path(__file__).resolve().parents[1] / "reports" / "figures"
reports_dir.mkdir(parents=True, exist_ok=True)

# 2. Plot Model Comparison
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
colors = ["#94a3b8", "#94a3b8", "#64748b", "#2563eb"]
bars = ax.bar(df_res["Model"], df_res["RMSLE"], color=colors, width=0.55)

ax.set_ylabel("RMSLE (Lower is Better)", fontsize=11)
ax.set_title("Validation Model Comparison (39-Day Horizon)", fontsize=13, fontweight="bold")
ax.set_ylim(0.45, 0.68)
ax.grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
out_path = reports_dir / "model_comparison.png"
plt.savefig(out_path)
print(f"Comparison plot saved to: {out_path}")