import json
import re
import matplotlib.pyplot as plt
from init import color_greedy, color_impr, color_evo, color_ats, color_ga

def plot_improved_and_evo_delta_over_weeks(
    infile: str,
    annotate: bool = True,
    band_half_factor: float = 0.125):
    """
    Plot Improved Greedy vs Evolutionary across weeks from an aggregated JSON:
      infile = runs/metrics_by_weeks.json (with a top-level "datasets" array)
    """

    bundle = json.load(open(infile, "r", encoding="utf-8"))

    datasets = bundle.get("datasets", [])
    if not datasets:
        raise ValueError("No 'datasets' found in the JSON.")

    # --- Determine weeks & order ---
    # Prefer explicit weeks_grid; fallback to parsing 'dataset' like 'week_12'
    weeks_grid = bundle.get("weeks_grid")
    def parse_week_label(lbl: str) -> int:
        m = re.search(r"(\d+)", lbl or "")
        if not m:
            raise ValueError(f"Could not parse week number from dataset label: {lbl}")
        return int(m.group(1))

    # Build a temp list with week numbers attached
    items = []
    for d in datasets:
        label = d.get("dataset", "")
        w = parse_week_label(label) if weeks_grid is None else None
        items.append({
            "label": label,
            "week": w,
            "improved": float(d["improved_greedy_metric"]),
            "evo": float(d["evo_best_metric"]),
        })

    # If weeks_grid exists, use its order; else sort by parsed week
    if weeks_grid:
        order = list(weeks_grid)
        # Map label->week via parsing if needed, then reorder by weeks_grid values
        # Some bundles may not include labels; in that case, assume order = weeks_grid
        # Try to pair each dataset to its week by matching the number in label
        mapped = []
        for w in order:
            # find the first item whose parsed week == w
            found = None
            for it in items:
                if it.get("week") is None:
                    # parse now if not parsed
                    it["week"] = parse_week_label(it["label"])
                if it["week"] == w:
                    found = it
                    break
            if found is None:
                raise ValueError(f"Week {w} from weeks_grid not found among dataset labels.")
            mapped.append(found)
        rows = mapped
    else:
        rows = sorted(items, key=lambda r: r["week"])

    weeks    = [r["week"]     for r in rows]
    improved = [r["improved"] for r in rows]
    evo      = [r["evo"]      for r in rows]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))

    # Lines on top
    evo_line, = ax.plot(weeks, evo,       marker='o', linewidth=2, color=color_evo, label="Proposed Method",    zorder=3)
    imp_line, = ax.plot(weeks, improved,  marker='o', linewidth=2, color=color_impr, label="Improved Greedy", zorder=3)

    # Shaded gap "pill" width
    if len(weeks) >= 2:
        step_min  = min(abs(b - a) for a, b in zip(weeks[:-1], weeks[1:]))
        band_half = band_half_factor * step_min
    else:
        band_half = 0.35

    # Minimal neutral gray shading behind lines
    band_color = "0.6"   # mid gray
    band_alpha = 0.12    # low opacity so it doesn't dominate

    y_all   = improved + evo
    y_min   = min(y_all)
    y_max   = max(y_all)
    y_range = max(1e-9, y_max - y_min)
    ax.set_ylim(y_min - 0.05*y_range, y_max + 0.12*y_range)

    # Shade each week's vertical difference
    for x, y_imp, y_evo in zip(weeks, improved, evo):
        y_low, y_high = sorted((y_imp, y_evo))
        ax.fill_between([x - band_half, x + band_half], y_low, y_high,
                        color=band_color, alpha=band_alpha, zorder=1)

    # Labels anchored to the LOWER (Improved) point
    if annotate:
        UMINUS = "−"
        for x, y_imp, y_evo in zip(weeks, improved, evo):
            ratio = (y_imp / y_evo * 100.0) if y_evo != 0 else float("nan")
            delta = y_imp - y_evo
            label = f"{ratio:.2f}% ({UMINUS}{abs(delta):.1f})" if delta < 0 else f"{ratio:.2f}% (+{delta:.1f})"
            ax.text(
                x, y_imp + 0.03*y_range, label,
                ha="center", va="bottom", fontsize=9, color="0.15",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="0.90", edgecolor="0.75", alpha=0.90),
                zorder=5
            )

    ax.set_title("Scheduling Metric by the number of Weeks (Improved Greedy vs Evolutionary)")
    ax.set_xlabel("Tournament length in weeks")
    ax.set_ylabel("Metric score")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()

    try:
        plt.show()
    except Exception:
        pass

def plot_top_of_ga_vs_ats_over_weeks(infile: str, annotate: bool = True,):

    bundle = json.load(open(infile, "r", encoding="utf-8"))
    datasets = bundle.get("datasets", [])
    if not datasets:
        raise ValueError("No 'datasets' found in the JSON.")

    # --- Determine week order (same logic as your original) ---
    weeks_grid = bundle.get("weeks_grid")

    def parse_week_label(lbl: str) -> int:
        m = re.search(r"(\d+)", lbl or "")
        if not m:
            raise ValueError(f"Could not parse week number from dataset label: {lbl}")
        return int(m.group(1))

    # Collect rows
    items = []
    for d in datasets:
        label = d.get("dataset", "")
        w = parse_week_label(label) if weeks_grid is None else None
        items.append({
            "label": label,
            "week": w,
            "ga":   float(d["ga_best_metric"]),
            "ats":  float(d["ats_best_metric"]),
            "evo":  float(d["evo_best_metric"]),
            "impr": float(d["improved_greedy_metric"]),
        })

    # Order by weeks_grid or by parsed week number
    if weeks_grid:
        order = list(weeks_grid)
        mapped = []
        for w in order:
            found = None
            for it in items:
                if it.get("week") is None:
                    it["week"] = parse_week_label(it["label"])
                if it["week"] == w:
                    found = it
                    break
            if found is None:
                raise ValueError(f"Week {w} from weeks_grid not found among dataset labels.")
            mapped.append(found)
        rows = mapped
    else:
        rows = sorted(items, key=lambda r: r["week"])

    weeks   = [r["week"] for r in rows]
    ga_vals = [r["ga"]   for r in rows]
    ats_vals= [r["ats"]  for r in rows]
    evo_vals= [r["evo"]  for r in rows]
    imp_vals= [r["impr"] for r in rows]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(weeks, evo_vals, marker='o', linewidth=2.2, color=color_evo,  label="Proposed Method")
    ax.plot(weeks, ats_vals, marker='o', linewidth=2.2, color=color_ats,  label="Adaptive Tabu Search")
    ax.plot(weeks, ga_vals,  marker='o', linewidth=2.2, color=color_ga,   label="Genetic Algorithm")
    ax.plot(weeks, imp_vals, marker='o', linewidth=2.2, color=color_impr, label="Improved Greedy")

    # --- Annotate all values on each series ---
    if annotate:
        def annotate_series(xs, ys, tag):
            for x, y in zip(xs, ys):
                ax.text(
                    x, y, f"{y:.1f}",
                    ha="center", va="bottom", fontsize=8.5, color="0.1",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="0.92", edgecolor="0.8", alpha=0.9)
                )
        annotate_series(weeks, ga_vals,  "GA")
        annotate_series(weeks, ats_vals, "ATS")
        annotate_series(weeks, evo_vals, "EVO")
        annotate_series(weeks, imp_vals, "IMPR")

    # Axes, grid, legend
    y_all   = ga_vals + ats_vals + evo_vals + imp_vals
    y_min   = min(y_all)
    y_max   = max(y_all)
    y_range = max(1e-9, y_max - y_min)
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.18 * y_range)

    ax.set_title("Metric Comparison by the number of Weeks")
    ax.set_xlabel("Tournament length in weeks")
    ax.set_ylabel("Metric score")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    infile = "./runs/metrics_by_weeks.json"
    plot_improved_and_evo_delta_over_weeks(infile, annotate=True)
    plot_top_of_ga_vs_ats_over_weeks(infile, annotate=True)
