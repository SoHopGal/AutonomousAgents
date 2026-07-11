import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# -------------------------------------------------------
# Load data
# -------------------------------------------------------

with open("fixed_runs.json", "r") as f:
    fixed_runs = json.load(f)

with open("adaptive_runs.json", "r") as f:
    adaptive_runs = json.load(f)

scenario_ids = [s["scenario"] for s in fixed_runs[0]]

summary = []

print("=" * 95)
print(
    f"{'Scenario':10}"
    f"{'Metric':15}"
    f"{'Fixed':18}"
    f"{'Adaptive':18}"
    f"{'Improve %':12}"
    f"{'p-value':12}"
)
print("=" * 95)

# -------------------------------------------------------
# Analyze every scenario
# -------------------------------------------------------

for scenario in scenario_ids:

    fixed_scenario = [
        run[[x["scenario"] for x in run].index(scenario)]
        for run in fixed_runs
    ]

    adaptive_scenario = [
        run[[x["scenario"] for x in run].index(scenario)]
        for run in adaptive_runs
    ]

    metrics = [
        ("avg_travel", "Travel Time"),
        ("avg_wait", "Waiting Time"),
        ("collisions", "Collisions")
    ]

    for metric_key, metric_name in metrics:

        fixed_values = np.array([x[metric_key] for x in fixed_scenario])
        adaptive_values = np.array([x[metric_key] for x in adaptive_scenario])

        fixed_mean = np.mean(fixed_values)
        adaptive_mean = np.mean(adaptive_values)

        fixed_std = np.std(fixed_values, ddof=1)
        adaptive_std = np.std(adaptive_values, ddof=1)

        fixed_ci = 1.96 * fixed_std / np.sqrt(len(fixed_values))
        adaptive_ci = 1.96 * adaptive_std / np.sqrt(len(adaptive_values))

        if fixed_mean != 0:
            improvement = (fixed_mean - adaptive_mean) / fixed_mean * 100
        else:
            improvement = 0

        t_stat, p_value = ttest_ind(
            fixed_values,
            adaptive_values,
            equal_var=False
        )

        print(
            f"{scenario:10}"
            f"{metric_name:15}"
            f"{fixed_mean:.2f} +/- {fixed_ci:.2f}".ljust(18)
            + f"{adaptive_mean:.2f} +/- {adaptive_ci:.2f}".ljust(18)
            + f"{improvement:8.2f}%   "
            + f"{p_value:.4f}"
        )

        summary.append({
            "Scenario": scenario,
            "Metric": metric_name,

            "Fixed Mean": fixed_mean,
            "Fixed Std": fixed_std,
            "Fixed CI95": fixed_ci,

            "Adaptive Mean": adaptive_mean,
            "Adaptive Std": adaptive_std,
            "Adaptive CI95": adaptive_ci,

            "Improvement %": improvement,

            "p-value": p_value
        })

    print("-" * 95)

# -------------------------------------------------------
# Save CSV
# -------------------------------------------------------

df = pd.DataFrame(summary)

df.to_csv("summary.csv", index=False)

print()
print("Saved summary.csv")
