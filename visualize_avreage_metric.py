import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


RUNS_DIR = Path(__file__).resolve().parent / "runs"
WEEK = 16

METRIC_FIELDS = [
    "improved_greedy_metric",
    "evo_best_metric",
    "ats_best_metric",
    "ga_best_metric",
    #"evo_time_elapsed",
    #"ats_time_elapsed",
    #"ga_time_elapsed",
]

LABELS = {
    "improved_greedy_metric": "Impr. Greedy",
    "evo_best_metric": "Proposed method",
    "ats_best_metric": "ATS",
    "ga_best_metric": "GA",
    #"evo_time_elapsed": "Evo_time",
    #"ats_time_elapsed": "ATS_time",
    #"ga_time_elapsed": "GA_time",
}

LABELS_IN_ORDER = [LABELS[f] for f in METRIC_FIELDS]

def _week_from_dataset_name(dataset_name: str) -> int | None:
    # fallback if weeks_grid missing or mismatched
    if isinstance(dataset_name, str) and dataset_name.startswith("week_"):
        try:
            return int(dataset_name.split("_", 1)[1])
        except Exception:
            return None
    return None


def extract_rows_from_file(json_path: Path, target_week: int | None) -> list[dict]:
    """
    Returns a list of rows. Each row corresponds to ONE dataset entry in the JSON file.
    Inclusion logic:
      - If weeks_grid exists and has an entry at index i, use weeks_grid[i] as the week.
      - Else fall back to parsing dataset["dataset"] like "week_16".
      - If target_week is not None, include only matching week entries.
    """
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    datasets = obj.get("datasets", [])
    weeks_grid = obj.get("weeks_grid", None)

    rows = []
    for i, d in enumerate(datasets):
        # Decide week for this dataset entry
        week_i = None
        if isinstance(weeks_grid, list) and i < len(weeks_grid):
            try:
                week_i = int(weeks_grid[i])
            except Exception:
                week_i = None

        if week_i is None:
            week_i = _week_from_dataset_name(d.get("dataset", ""))

        # Filter
        if target_week is not None and week_i != target_week:
            continue

        row = {
            "path": str(json_path),
            "file_run": json_path.parent.name,  # folder name
            "entry_idx": i,                      # which dataset entry inside file
            "week": week_i,
            "timestamp": obj.get("timestamp"),
            "schema_version": obj.get("schema_version"),
        }

        for k in METRIC_FIELDS:
            row[k] = d.get(k, np.nan)

        rows.append(row)

    return rows


def collect_all_entries(runs_dir: Path, target_week: int | None) -> pd.DataFrame:
    files = sorted(runs_dir.rglob("metrics_by_weeks.json"))
    all_rows = []
    for fp in files:
        all_rows.extend(extract_rows_from_file(fp, target_week=target_week))

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    for k in METRIC_FIELDS:
        df[k] = pd.to_numeric(df[k], errors="coerce")

    return df

if __name__ == "__main__":
    df_runs = collect_all_entries(RUNS_DIR, target_week=WEEK)
    if df_runs.empty:
        raise RuntimeError(
            f"No dataset entries found under {RUNS_DIR} matching week={WEEK}."
        )

    data = []
    for field in METRIC_FIELDS:
        data.append(df_runs[field].dropna().tolist())

    labels = LABELS_IN_ORDER

    print(f"Number of runs: {len(data[0])}")

    stats = {
        "Min": [np.min(d) for d in data],
        "Max": [np.max(d) for d in data],
        "Mean": [np.mean(d) for d in data],
        "Range": [np.ptp(d) for d in data],
        "Std. Dev.": [np.std(d, ddof=0) for d in data],
    }

    print("\\textbf{Statistic} & \\textbf{ImprovedGreedy} & \\textbf{Proposed Method} & \\textbf{ATS} & \\textbf{GA} \\\\")
    for stat, values in stats.items():
        vals = [round(v, 1) for v in values]
        max_idx = int(np.argmax(vals))  # best (highest) value (same as your old code)
        formatted = " & ".join(
            [f"\\textbf{{{v}}}" if i == max_idx else f"{v}" for i, v in enumerate(vals)]
        )
        print(f"{stat:<11}& {formatted} \\\\")

    plt.figure(figsize=(8, 5))
    parts = plt.violinplot(data, showmeans=True, showextrema=True, showmedians=True)

    for pc in parts["bodies"]:
        pc.set_facecolor("#87A7EB")  # light blue
        pc.set_edgecolor("black")
        pc.set_alpha(0.8)

    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("Resulting Metric Value [-]")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    wide = pd.DataFrame(
        {
            "Improved Greedy": data[0],
            "Proposed Method": data[1],
            "ATS": data[2],
            "GA": data[3],
        }
    )

    df_long = wide.melt(var_name="Method", value_name="Metric")

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        x="Method",
        y="Metric",
        data=df_long,
        width=0.5,
        fliersize=0,
        color="#87A7EB",
        boxprops={"alpha": 0.7},
    )
    sns.swarmplot(
        x="Method",
        y="Metric",
        data=df_long,
        color="black",
        alpha=0.7,
        size=5,
    )

    plt.title("Comparison of Metric Values Across Methods", fontsize=14)
    plt.xlabel("")
    plt.ylabel("Metric Value")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()