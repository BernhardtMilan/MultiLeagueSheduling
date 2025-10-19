import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- config ----------
PATH_GREEDY   = "./runs/per_team_metric_GREEDY.json"
PATH_IMPROVED = "./runs/per_team_metric_IMPROVED GREEDY.json"
PATH_EVO      = "./runs/per_team_metric_EVOLUTIONARY.json"
BOTTOM_K = 5

# ---------- load ----------
with open(PATH_GREEDY, "r", encoding="utf-8") as f:
    raw_g = json.load(f)
with open(PATH_IMPROVED, "r", encoding="utf-8") as f:
    raw_i = json.load(f)
with open(PATH_EVO, "r", encoding="utf-8") as f:
    raw_e = json.load(f)

def dict_to_df(d, method_name):
    df = pd.DataFrame.from_dict(d, orient="index")
    df.index.name = "team"
    df.reset_index(inplace=True)
    df["method"] = method_name
    for col in ["availability", "match_bunching_penalty", "idle_gap_penalty", "spread_reward"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df_g = dict_to_df(raw_g, "GREEDY")
df_i = dict_to_df(raw_i, "IMPROVED GREEDY")
df_e = dict_to_df(raw_e, "EVOLUTIONARY")
df   = pd.concat([df_g, df_i, df_e], ignore_index=True)

# shared component metrics across ALL THREE
base_candidates = ["availability", "match_bunching_penalty", "idle_gap_penalty", "spread_reward"]
base_metrics = [m for m in base_candidates if all(m in d.columns for d in (df_g, df_i, df_e))]

# ---- add OVERALL back: sum of already-weighted components ----
for dfx in (df_g, df_i, df_e):
    dfx["overall"] = dfx[base_metrics].sum(axis=1, skipna=True)
df = pd.concat([df_g, df_i, df_e], ignore_index=True)

# ---------- stdout summaries ----------
metrics_for_tables = base_metrics + ["overall"]

avg = df.groupby("method")[metrics_for_tables].mean().round(3)
print("=== Per-metric averages (components + overall) ===")
print(avg.to_string(), "\n")

def bottom_k(df_, metric, k=BOTTOM_K):
    return (df_[["team","method",metric]]
            .sort_values(metric, ascending=True)
            .head(k))

print("=== Bottom-K teams per metric (lowest values) ===")
for m in metrics_for_tables:
    print(f"\n-- {m} --")
    print(bottom_k(df, m).to_string(index=False))

def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5*iqr
    hi = q3 + 1.5*iqr
    return (series < lo) | (series > hi), lo, hi

print("\n=== IQR outliers by method & metric ===")
any_outliers = False
for method, g in df.groupby("method"):
    for m in metrics_for_tables:
        s = g[m].dropna()
        mask, lo, hi = iqr_outliers(s)
        out = g.loc[s.index[mask], ["team", m]].sort_values(m)
        if not out.empty:
            any_outliers = True
            print(f"\n[{method}] {m} outliers (outside [{lo:.2f}, {hi:.2f}]):")
            print(out.to_string(index=False))
if not any_outliers:
    print("No outliers found via IQR.")

# ---------- plotting helpers ----------
def _prep_constant_noise(v):
    """Add tiny noise if a series is constant so violins don’t collapse."""
    if v.size and np.allclose(v, v[0]):
        v = v.astype(float) + np.random.default_rng(0).normal(0, 1e-6, size=v.size)
    return v

def strip_grid(metrics, sharey=False):
    """One row, len(metrics) columns; GREEDY/IMPROVED/EVO as jittered strips per subplot."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.8*n, 4.8), sharey=sharey)
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, m in zip(axes, metrics):
        g_vals = df_g[m].dropna().values
        i_vals = df_i[m].dropna().values
        e_vals = df_e[m].dropna().values

        jg = np.zeros(g_vals.size) + rng.normal(0, 0.04, size=g_vals.size)
        ji = np.ones(i_vals.size)  + rng.normal(0, 0.04, size=i_vals.size)
        je = np.full(e_vals.size, 2.0) + rng.normal(0, 0.04, size=e_vals.size)

        ax.scatter(jg, g_vals, alpha=0.6, s=20, marker="o", label="GREEDY")
        ax.scatter(ji, i_vals, alpha=0.6, s=20, marker="o", label="IMPROVED GREEDY")
        ax.scatter(je, e_vals, alpha=0.6, s=20, marker="o", label="EVOLUTIONARY")

        ax.set_xticks([0, 1, 2], ["GREEDY", "IMPROVED", "EVO"])
        ax.set_title(m.replace("_", " "))
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("score")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, ["GREEDY", "IMPROVED GREEDY", "EVOLUTIONARY"], loc="upper center", ncol=3)
    fig.suptitle("Components — strip plots", y=1, fontsize=12)
    fig.tight_layout()

def violin_grid(metrics, sharey=False):
    """One row, len(metrics) columns; three side-by-side violins per subplot."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.8*n, 4.8), sharey=sharey)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        g_vals = _prep_constant_noise(df_g[m].dropna().values)
        i_vals = _prep_constant_noise(df_i[m].dropna().values)
        e_vals = _prep_constant_noise(df_e[m].dropna().values)
        ax.violinplot([g_vals, i_vals, e_vals], positions=[0, 1, 2],
                      showmeans=True, showextrema=True, showmedians=True)
        ax.set_xticks([0, 1, 2], ["GREEDY", "IMPROVED", "EVO"])
        ax.set_title(m.replace("_", " "))
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("score")
    fig.suptitle("Components + OVERALL — violin plots", fontsize=12)
    fig.tight_layout()

strip_grid(base_metrics, sharey=False)
strip_grid(["overall"], sharey=True)
violin_grid(base_metrics, sharey=False)
violin_grid(["overall"], sharey=True)


plt.show()
