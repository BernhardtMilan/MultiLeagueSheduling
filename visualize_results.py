import json
import math
import matplotlib.pyplot as plt
import numpy as np

file1 = "./runs/metrics_by_directory.json"
file2 = "./runs/metrics_by_weeks.json"

data = json.load(open(file2,"r"))

datasets = [d["dataset"] for d in data["datasets"]]
n = len(datasets)
x = np.arange(n)

def print_summary_table(ds):
    # Map algorithms -> (total_metric_field, component_array_field)
    cols = [
        ("Greedy",          "greedy_metric",          "greedy_scores_weighted"),
        ("ImprovedGreedy",  "improved_greedy_metric", "improved_scores_weighted"),
        ("Mine",            "evo_best_metric",        "evo_scores_weighted"),  # "Mine" = your Evo
        ("ATS",             "ats_best_metric",        "ats_scores_weighted"),
        ("GA",              "ga_best_metric",         "ga_scores_weighted"),
    ]

    # Build rows
    headers = [c[0] for c in cols]
    total_vals = [ds[c[1]] for c in cols]
    comp_vals = [ds[c[2]] for c in cols]  # arrays: [availability, bunching, gaps, spread, L1]

    rows = [
        ("Total metric", total_vals),
        ("Availability", [v[0] for v in comp_vals]),
        ("Bunching",     [v[1] for v in comp_vals]),
        ("Gaps",         [v[2] for v in comp_vals]),
        ("Spread",       [v[3] for v in comp_vals]),
        ("L1 pitch",     [v[4] for v in comp_vals]),
    ]

    # Helpers for formatting
    def best_index(values):
        # Higher is better for all these rows (penalties are negative, so "closest to 0" = max)
        return int(max(range(len(values)), key=lambda i: values[i]))

    def fmt_num(x):
        # The JSON mixes ints/floats; show as int when close, otherwise 1 decimal
        return f"{int(round(x))}" if abs(x - round(x)) < 1e-6 else f"{x:.1f}"

    # Column widths
    name_w = 12
    col_w = 14

    # Header
    print()
    title = f"Dataset: {ds.get('dataset','(unknown)')}"
    print(title)
    print("-" * (name_w + col_w*len(headers)))
    print(f"{'Metric':<{name_w}}", end="")
    for h in headers:
        print(f"{h:>{col_w}}", end="")
    print()

    # Body (bold-like marker • on the best value per row)
    for rname, vals in rows:
        b = best_index(vals)
        print(f"{rname:<{name_w}}", end="")
        for i, v in enumerate(vals):
            s = fmt_num(v)
            # Mark best with a centered dot; easy to spot in mono output
            s = f"{s} •" if i == b else s
            print(f"{s:>{col_w}}", end="")
        print()
    print("-" * (name_w + col_w*len(headers)))
    print()

for ds in data["datasets"]:
    print_summary_table(ds)

# ---- Figure 1: Overall metrics per dataset (grouped bars) ----
greedy_metrics   = [d["greedy_metric"] for d in data["datasets"]]
improv_metrics   = [d["improved_greedy_metric"] for d in data["datasets"]]
evo_metrics      = [d["evo_best_metric"] for d in data["datasets"]]
ats_metrics      = [d["ats_best_metric"] for d in data["datasets"]]
ga_metrics       = [d["ga_best_metric"] for d in data["datasets"]]

width = 0.2
plt.figure(figsize=(10, 5))
plt.bar(x - 2*width, greedy_metrics,  width, label="Greedy")
plt.bar(x - 1*width, improv_metrics,  width, label="Improved Greedy")
plt.bar(x, evo_metrics,     width, label="Evo Best")
plt.bar(x + 1*width, ats_metrics,     width, label="ATS")
plt.bar(x + 2*width, ga_metrics,      width, label="GA")

plt.xticks(x, datasets, rotation=15)
plt.ylabel("Metric")
plt.title("Schedule quality metrics across datasets")
plt.legend()
plt.tight_layout()

all_vals = greedy_metrics + improv_metrics + evo_metrics + ats_metrics + ga_metrics
ymin = 15000
ymax = max(all_vals) * 1.02  # tiny headroom
plt.ylim(ymin, ymax)

# ---- Figure 2: Availability buckets for evo (stacked bars) ----
buckets = data["availability_buckets"]  # ["bad","no_answer","might","good"]
segments = list(zip(*[d["evo_value_counts"] for d in data["datasets"]]))  # shape (4, n)

plt.figure(figsize=(10, 5))
bottom = np.zeros(n, dtype=float)
for i, seg in enumerate(segments):
    plt.bar(x, seg, bottom=bottom, label=buckets[i])
    bottom += np.array(seg, dtype=float)

plt.xticks(x, datasets, rotation=15)
plt.ylabel("Count")
plt.title("Evo schedule availability distribution by dataset")
plt.legend(title="Bucket")
plt.tight_layout()

plt.show()
