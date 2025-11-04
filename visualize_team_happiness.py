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

# ---------- helpers ----------
BASE_COLS = ["availability", "match_bunching_penalty", "idle_gap_penalty", "spread_reward"]

def dict_to_df(d, method_name):
    df = pd.DataFrame.from_dict(d, orient="index")
    df.index.name = "team"
    df.reset_index(inplace=True)
    df["method"] = method_name
    # ensure numeric
    for col in BASE_COLS + ["number_of_matches"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    return df

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def norm_linear(x, lo, hi):
    # safe min-max to [0,1]
    return clamp01((x - lo) / (hi - lo)) if (hi is not None and lo is not None and hi > lo) else np.zeros_like(x, dtype=float)

def norm_penalty_to_pct(penalty_values, worst_negative):
    """
    penalty_values ≤ 0 are better when closer to 0.
    worst_negative is the most negative allowed (e.g., -3*(n-1)).
    Map: 0 -> 100%, worst_negative -> 0%.
    """
    worst_mag = np.maximum(np.abs(worst_negative), 1e-9)  # avoid zero
    frac = 1.0 - (np.abs(penalty_values) / worst_mag)
    return clamp01(frac)

def compute_component_bounds(n_matches):
    """
    Return:
      - availability min/max (weighted)
      - spread min/max (weighted)
      - worst bunching magnitude (positive)
      - worst idle magnitude (positive)
    According to your rules:
      availability: ±80 for 10 matches, ±88 for 11
      spread: [0.2, 2.0] for 10; [0.2, 2.2] for 11
      bunching worst: 3*(n-1)   (because weight = -3)
      idle    worst: 1*(n-1)    (because weight = -1)
    Fallback is linear if n not 10/11.
    """
    if n_matches == 10:
        a_lo, a_hi = -80.0,  80.0
        s_lo, s_hi =   0.2,   2.0
    elif n_matches == 11:
        a_lo, a_hi = -88.0,  88.0
        s_lo, s_hi =   0.2,   2.2
    else:
        # Fallback: use 8 points per match for availability, 0.2 per played week for spread.
        per_match_avail = 8.0
        a_lo, a_hi = -per_match_avail * n_matches, per_match_avail * n_matches
        s_lo, s_hi = 0.2, 0.2 * n_matches
    worst_bunch = 3.0 * max(n_matches - 1, 0)  # weight -3
    worst_idle  = 1.0 * max(n_matches - 1, 0)  # weight -1
    return a_lo, a_hi, s_lo, s_hi, worst_bunch, worst_idle

def infer_matches_if_missing(row):
    """
    If 'number_of_matches' missing, estimate conservatively from spread_reward or availability:
      - prefer spread_reward: n ≈ round(sr / 0.2), clipped to [8, 14]
      - else use availability magnitude vs 8 points/match: n ≈ round(|a| / 8), clipped
    """
    n = row.get("number_of_matches")
    if pd.notna(n) and n > 0:
        return int(n)
    sr = row.get("spread_reward", np.nan)
    a  = row.get("availability", np.nan)
    if pd.notna(sr):
        est = int(np.clip(round(sr / 0.2), 8, 14))
        return est
    if pd.notna(a):
        est = int(np.clip(round(abs(a) / 8.0), 8, 14))
        return est
    return 10  # safe default

def add_overall_and_normalized(df):
    # overall = sum of weighted components (already-weighted inputs)
    df["overall"] = df[BASE_COLS].sum(axis=1, skipna=True)

    # allocate outputs
    for name in ["availability_pct", "match_bunching_penalty_pct", "idle_gap_penalty_pct", "spread_reward_pct", "overall_pct"]:
        df[name] = np.nan

    for idx, row in df.iterrows():
        n_m = infer_matches_if_missing(row)
        a   = float(row.get("availability") or 0.0)
        sr  = float(row.get("spread_reward") or 0.0)
        bp  = float(row.get("match_bunching_penalty") or 0.0)
        ig  = float(row.get("idle_gap_penalty") or 0.0)
        ov  = float(a + sr + bp + ig)

        a_lo, a_hi, s_lo, s_hi, worst_b, worst_i = compute_component_bounds(n_m)

        a_pct = norm_linear(a,  a_lo, a_hi)
        s_pct = norm_linear(sr, s_lo, s_hi)
        b_pct = norm_penalty_to_pct(bp, -worst_b)
        i_pct = norm_penalty_to_pct(ig, -worst_i)

        # overall bounds = sum of component bounds (worst penalties are negative)
        ov_min = a_lo + (-worst_b) + (-worst_i) + s_lo
        ov_max = a_hi + 0.0        + 0.0        + s_hi
        ov_pct = norm_linear(ov, ov_min, ov_max)

        df.at[idx, "availability_pct"]           = a_pct * 100.0
        df.at[idx, "spread_reward_pct"]          = s_pct * 100.0
        df.at[idx, "match_bunching_penalty_pct"] = b_pct * 100.0
        df.at[idx, "idle_gap_penalty_pct"]       = i_pct * 100.0
        df.at[idx, "overall_pct"]                = ov_pct * 100.0

    return df

# ---------- build dataframes ----------
df_g  = add_overall_and_normalized(dict_to_df(raw_g,  "GREEDY"))
df_i  = add_overall_and_normalized(dict_to_df(raw_i,  "IMPROVED GREEDY"))
df_e  = add_overall_and_normalized(dict_to_df(raw_e,  "EVOLUTIONARY"))
df_a  = add_overall_and_normalized(dict_to_df(raw_a,  "ATS"))
df_ga = add_overall_and_normalized(dict_to_df(raw_ga, "GA"))

df    = pd.concat([df_g, df_i, df_e, df_a, df_ga], ignore_index=True)

# shared component metrics across all methods (raw)
base_metrics = [m for m in BASE_COLS if all(m in d.columns for d in (df_g, df_i, df_e, df_a, df_ga))]
metrics_for_tables = base_metrics + ["overall"]

# ---------- stdout summaries (raw) ----------
avg_raw = df.groupby("method")[metrics_for_tables].mean().round(3)
print("=== RAW per-metric averages (components + overall) ===")
print(avg_raw.to_string(), "\n")

# ---------- stdout summaries (normalized 0–100%) ----------
pct_cols = ["availability_pct", "match_bunching_penalty_pct", "idle_gap_penalty_pct", "spread_reward_pct", "overall_pct"]
avg_pct = df.groupby("method")[pct_cols].mean().round(2)
print("=== NORMALIZED per-metric averages (0–100%) ===")
print(avg_pct.to_string(), "\n")

def bottom_k(df_, metric, k=BOTTOM_K):
    return (df_[["team","method",metric]]
            .sort_values(metric, ascending=True)
            .head(k))

print("=== Bottom-K teams per metric (lowest % values) ===")
for m in pct_cols:
    print(f"\n-- {m} --")
    print(bottom_k(df, m).to_string(index=False))

# ---------- plotting helpers ----------
def _prep_constant_noise(v):
    """Add tiny noise if a series is constant so violins don’t collapse."""
    if v.size and np.allclose(v, v[0]):
        v = v.astype(float) + np.random.default_rng(0).normal(0, 1e-6, size=v.size)
    return v

def strip_grid(metrics_pct, start=0, sharey=True):
    """One row, len(metrics_pct) columns; five methods as jittered strips per subplot."""
    n = len(metrics_pct)
    fig, axes = plt.subplots(1, n, figsize=(4.8*n, 4.8), sharey=sharey)
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    x_pos = {"GREEDY":0.0, "IMPROVED GREEDY":1.0, "EVOLUTIONARY":2.0, "ATS":3.0, "GA":4.0}

    for ax, m in zip(axes, metrics_pct):
        g_vals  = df_g[m].dropna().values
        i_vals  = df_i[m].dropna().values
        e_vals  = df_e[m].dropna().values
        a_vals  = df_a[m].dropna().values
        ga_vals = df_ga[m].dropna().values

        jitter = lambda n: rng.normal(0, 0.06, size=n)

        ax.scatter(x_pos["GREEDY"]          + jitter(g_vals.size),  g_vals,  alpha=0.6, s=20, marker="o", label="GREEDY")
        ax.scatter(x_pos["IMPROVED GREEDY"] + jitter(i_vals.size),  i_vals,  alpha=0.6, s=20, marker="o", label="IMPR. GREEDY")
        ax.scatter(x_pos["EVOLUTIONARY"]    + jitter(e_vals.size),  e_vals,  alpha=0.6, s=20, marker="o", label="EVOLUTIONARY")
        ax.scatter(x_pos["ATS"]             + jitter(a_vals.size),  a_vals,  alpha=0.6, s=20, marker="o", label="ATS")
        ax.scatter(x_pos["GA"]              + jitter(ga_vals.size), ga_vals, alpha=0.6, s=20, marker="o", label="GA")

        # Keep positions but remove labels to avoid duplicates with legend
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels([])

        ax.set_ylim(start, 101)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel(f"team happines (%)")

    # Reserve space for bottom legend and top title, then layout
    fig.tight_layout(rect=[0, 0.14, 1, 0.90])

    # Single legend across the bottom (keeps marker colors/styles)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:5], labels[:5],
        loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, 0.03),
        frameon=False,
        handletextpad=0.1,   # distance between marker and text
        columnspacing=0.8,   # spacing between legend columns
        borderpad=0.3,       # padding inside legend box
    )

    ax.set_title("Metric of teams using the different algorithms")

def violin_grid(metrics_pct, start=0, sharey=True):
    """One row, len(metrics_pct) columns; five side-by-side violins per subplot."""
    n = len(metrics_pct)
    fig, axes = plt.subplots(1, n, figsize=(4.8*n, 4.8), sharey=sharey)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics_pct):
        g_vals  = _prep_constant_noise(df_g[m].dropna().values)
        i_vals  = _prep_constant_noise(df_i[m].dropna().values)
        e_vals  = _prep_constant_noise(df_e[m].dropna().values)
        a_vals  = _prep_constant_noise(df_a[m].dropna().values)
        ga_vals = _prep_constant_noise(df_ga[m].dropna().values)

        ax.violinplot([g_vals, i_vals, e_vals, a_vals, ga_vals],
                      positions=[0,1,2,3,4],
                      showmeans=True, showextrema=True, showmedians=True)
        ax.set_xticks([0,1,2,3,4], ["GREEDY", "IMPROVED", "EVO", "ATS", "GA"])
        ax.set_ylim(start, 100)
        ax.set_title(m.replace("_pct","").replace("_"," ").title())
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel(f"percent ({start}-100)")
    fig.suptitle("TITLE KELL MÉG", fontsize=12)
    fig.tight_layout()

# ---------- build lists for plotting ----------
components_pct = ["availability_pct", "match_bunching_penalty_pct", "idle_gap_penalty_pct", "spread_reward_pct"]
overall_pct    = ["overall_pct"]

# ---------- make composite figures ----------

strip_grid(overall_pct,    start=88, sharey=True)
#violin_grid(components_pct, start=70, sharey=True)


#for start in [0, 70]:
#    strip_grid(components_pct, start=start, sharey=True, title=f"Components — strip ({start}-100%)")
#    strip_grid(overall_pct,    start=start, sharey=True, title=f"OVERALL — strip ({start}-100%)")

#    violin_grid(components_pct, start=start, sharey=True, title=f"Components — violin ({start}-100%)")
#    violin_grid(overall_pct,    start=start, sharey=True, title=f"OVERALL — violin ({start}-100%)")

plt.show()
