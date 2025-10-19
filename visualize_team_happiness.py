import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- config ----------
PATH_GREEDY   = "./runs/per_team_metric_GREEDY.json"
PATH_IMPROVED = "./runs/per_team_metric_IMPROVED GREEDY.json"
PATH_EVO      = "./runs/per_team_metric_EVOLUTIONARY.json"
PATH_ATS      = "./runs/per_team_metric_ATS.json"
PATH_GA       = "./runs/per_team_metric_GA.json"
BOTTOM_K = 5

# ---------- load ----------
with open(PATH_GREEDY, "r", encoding="utf-8") as f:
    raw_g = json.load(f)
with open(PATH_IMPROVED, "r", encoding="utf-8") as f:
    raw_i = json.load(f)
with open(PATH_EVO, "r", encoding="utf-8") as f:
    raw_e = json.load(f)
with open(PATH_ATS, "r", encoding="utf-8") as f:
    raw_a = json.load(f)
with open(PATH_GA, "r", encoding="utf-8") as f:
    raw_ga = json.load(f)

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
df_a = dict_to_df(raw_a, "ATS")
df_ga = dict_to_df(raw_ga, "GA")
df   = pd.concat([df_g, df_i, df_e, df_a, df_ga], ignore_index=True)

# shared component metrics across all methods
base_candidates = ["availability", "match_bunching_penalty", "idle_gap_penalty", "spread_reward"]
base_metrics = [m for m in base_candidates if all(m in d.columns for d in (df_g, df_i, df_e, df_a, df_ga))]

# ---- add OVERALL back: sum of already-weighted components ----
for dfx in (df_g, df_i, df_e, df_a, df_ga):
    dfx["overall"] = dfx[base_metrics].sum(axis=1, skipna=True)
df = pd.concat([df_g, df_i, df_e, df_a, df_ga], ignore_index=True)

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
    """One row, len(metrics) columns; GREEDY / IMPROVED GREEDY / EVO / ATS / GA as jittered strips per subplot."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.8*n, 4.8), sharey=sharey)
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    x_pos = {
        "GREEDY": 0.0,
        "IMPROVED GREEDY": 1.0,
        "EVOLUTIONARY": 2.0,
        "ATS": 3.0,
        "GA": 4.0,
    }

    for ax, m in zip(axes, metrics):
        g_vals  = df_g[m].dropna().values
        i_vals  = df_i[m].dropna().values
        e_vals  = df_e[m].dropna().values
        a_vals  = df_a[m].dropna().values
        ga_vals = df_ga[m].dropna().values

        jitter = lambda n: rng.normal(0, 0.06, size=n)

        ax.scatter(x_pos["GREEDY"]            + jitter(g_vals.size),  g_vals,  alpha=0.6, s=20, marker="o", label="GREEDY")
        ax.scatter(x_pos["IMPROVED GREEDY"]   + jitter(i_vals.size),  i_vals,  alpha=0.6, s=20, marker="o", label="IMPROVED GREEDY")
        ax.scatter(x_pos["EVOLUTIONARY"]      + jitter(e_vals.size),  e_vals,  alpha=0.6, s=20, marker="o", label="EVOLUTIONARY")
        ax.scatter(x_pos["ATS"]               + jitter(a_vals.size),  a_vals,  alpha=0.6, s=20, marker="o", label="ATS")
        ax.scatter(x_pos["GA"]                + jitter(ga_vals.size), ga_vals, alpha=0.6, s=20, marker="o", label="GA")

        ax.set_xticks([0, 1, 2, 3, 4], ["GREEDY", "IMPROVED", "EVO", "ATS", "GA"])
        ax.set_title(m.replace("_", " "))
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("score")
    # Single shared legend with 5 entries
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles[:5], ["GREEDY", "IMPROVED GREEDY", "EVOLUTIONARY", "ATS", "GA"],
               loc="upper center", ncol=5)
    fig.suptitle("Components — strip plots", y=1.02, fontsize=12)
    fig.tight_layout()

def violin_grid(metrics, sharey=False):
    """One row, len(metrics) columns; five side-by-side violins per subplot."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4.8*n, 4.8), sharey=sharey)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        g_vals  = _prep_constant_noise(df_g[m].dropna().values)
        i_vals  = _prep_constant_noise(df_i[m].dropna().values)
        e_vals  = _prep_constant_noise(df_e[m].dropna().values)
        a_vals  = _prep_constant_noise(df_a[m].dropna().values)
        ga_vals = _prep_constant_noise(df_ga[m].dropna().values)

        parts = ax.violinplot([g_vals, i_vals, e_vals, a_vals, ga_vals],
                              positions=[0, 1, 2, 3, 4],
                              showmeans=True, showextrema=True, showmedians=True)
        ax.set_xticks([0, 1, 2, 3, 4], ["GREEDY", "IMPROVED", "EVO", "ATS", "GA"])
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
