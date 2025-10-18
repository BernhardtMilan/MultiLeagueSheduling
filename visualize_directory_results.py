import json
import math
import matplotlib.pyplot as plt
import numpy as np

data = json.load(open("./runs/metrics_by_dataset.json","r"))

datasets = [d["dataset"] for d in data["datasets"]]
n = len(datasets)
x = np.arange(n)

# ---- Figure 1: Overall metrics per dataset (grouped bars) ----
random_metrics   = [d["random_metric"] for d in data["datasets"]]
greedy_metrics   = [d["greedy_metric"] for d in data["datasets"]]
improv_metrics   = [d["improved_greedy_metric"] for d in data["datasets"]]
evo_metrics      = [d["evo_best_metric"] for d in data["datasets"]]

width = 0.2
plt.figure(figsize=(10, 5))
plt.bar(x - 1.5*width, random_metrics,  width, label="Random")
plt.bar(x - 0.5*width, greedy_metrics,  width, label="Greedy")
plt.bar(x + 0.5*width, improv_metrics,  width, label="Improved Greedy")
plt.bar(x + 1.5*width, evo_metrics,     width, label="Evo Best")

plt.xticks(x, datasets, rotation=15)
plt.ylabel("Metric")
plt.title("Schedule quality metrics across datasets")
plt.legend()
plt.tight_layout()

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
