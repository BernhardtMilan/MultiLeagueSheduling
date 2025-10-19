import os
import sys
import json
import subprocess
from pathlib import Path
import contextlib

WEEK_GRID = [16]
#WEEK_GRID = [10, 11, 12, 14, 16]

SELF = Path(__file__).resolve()

def compute_once() -> dict:
    """
    Runs one full experiment for the CURRENT process using SORSOLO_WEEKS
    (which must already be set in the environment).
    """
    # Delayed imports so init.py reads the env var at import time
    from init import weights
    from evolutionary import evolutionary
    from non_ai_sorts import run_non_ai_sorts
    from sorsolo import calulate_team_metric

    def get_weighted_scores(scores):
        return [
            round(weights["availability"]           * scores[0], 4),
            round(weights["match_bunching_penalty"] * scores[1], 4),
            round(weights["idle_gap_penalty"]       * scores[2], 4),
            round(weights["spread_reward"]          * scores[3], 4),
            round(weights["L1_pitch_penalty"]       * scores[4], 4),
        ]

    (
        team_schedules,
        possible_max_metric,
        league_teams,
        random_draw_data,
        greedy_draw_data,
        impoved_greedy_draw_data,
    ) = run_non_ai_sorts()

    evo_best_metric, evo_best_draw_structure, evo_best_scores, evo_best_value_counts = evolutionary(
        impoved_greedy_draw_data[0],
        team_schedules,
        possible_max_metric,
        league_teams,
        plot=False
    )

    METHODS_AND_STRUCTS = [
        ("RANDOM",           random_draw_data[0]),
        ("GREEDY",           greedy_draw_data[0]),
        ("IMPROVED GREEDY",  impoved_greedy_draw_data[0]),
        ("EVOLUTIONARY",     evo_best_draw_structure),
    ]

    for method, draw_structure in METHODS_AND_STRUCTS:
        per_team_metric = calulate_team_metric(draw_structure, team_schedules, league_teams)
        out_path = os.path.join("runs", f"per_team_metric_{method}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(per_team_metric, f, indent=2, ensure_ascii=False)

    return {
        "random_metric": float(random_draw_data[1]),
        "greedy_metric": float(greedy_draw_data[1]),
        "improved_greedy_metric": float(impoved_greedy_draw_data[1]),
        "evo_best_metric": float(evo_best_metric),

        "random_scores_weighted":   get_weighted_scores(list(random_draw_data[2])),
        "greedy_scores_weighted":   get_weighted_scores(list(greedy_draw_data[2])),
        "improved_scores_weighted": get_weighted_scores(list(impoved_greedy_draw_data[2])),
        "evo_scores_weighted":      get_weighted_scores(list(evo_best_scores)),

        "random_value_counts":   list(random_draw_data[3]),
        "greedy_value_counts":   list(greedy_draw_data[3]),
        "improved_value_counts": list(impoved_greedy_draw_data[3]),
        "evo_value_counts":      list(evo_best_value_counts),
    }

def print_human(res: dict):
    print("--------Metrics--------")
    print(f"Random: {res['random_metric']}")
    print(f"Greedy: {res['greedy_metric']}")
    print(f"Improved Greedy: {res['improved_greedy_metric']}")
    print(f"Evolutionary: {res['evo_best_metric']}\n")
    print("Scores:")
    print("[total_availability_score, bunching_penalty, idle_gap_penalty, spread_reward, L1_pitch_penalty]")
    print(res["random_scores_weighted"])
    print(res["greedy_scores_weighted"])
    print(res["improved_scores_weighted"])
    print(res["evo_scores_weighted"])
    print("\nMatch availability:")
    print("[bad, no answer, might, good]")
    print(res["random_value_counts"])
    print(res["greedy_value_counts"])
    print(res["improved_value_counts"])
    print(res["evo_value_counts"])

def run_child_with_weeks(weeks: int) -> dict:
    """
    Spawn a fresh process of this same file in --child mode with SORSOLO_WEEKS set.
    Stream child's STDERR live (so you see its prints/logs), and collect JSON from STDOUT.
    """
    import threading

    env = os.environ.copy()
    env["SORSOLO_WEEKS"] = str(weeks)
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("PYTHONUNBUFFERED", "1")  # helps streaming on Windows

    # -u = unbuffered stdio so logs stream immediately
    proc = subprocess.Popen(
        [sys.executable, "-u", str(SELF), "--child"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # line-buffered
    )

    stdout_lines: list[str] = []

    # pump stdout into a buffer (do NOT print; it's the JSON)
    def pump_stdout():
        try:
            for line in iter(proc.stdout.readline, ''):
                stdout_lines.append(line)
        except ValueError:
            # pipe closed while iterating; safe to ignore
            pass

    # pump stderr to our stdout LIVE, prefixed with the week
    def pump_stderr():
        try:
            for line in iter(proc.stderr.readline, ''):
                sys.stdout.write(f"[week {weeks}] {line}")
                sys.stdout.flush()
        except ValueError:
            # pipe closed while iterating; safe to ignore
            pass

    t_out = threading.Thread(target=pump_stdout, daemon=True)
    t_err = threading.Thread(target=pump_stderr, daemon=True)
    t_out.start()
    t_err.start()

    # wait for process to finish, then join threads
    proc.wait()
    t_out.join()
    t_err.join()

    # close pipes explicitly
    if proc.stdout:
        try: proc.stdout.close()
        except: pass
    if proc.stderr:
        try: proc.stderr.close()
        except: pass

    if proc.returncode != 0:
        raise RuntimeError(f"Run failed for weeks={weeks} (exit {proc.returncode}). See streamed logs above.")

    out = ''.join(stdout_lines).strip()
    if not out:
        raise RuntimeError(f"Child produced empty stdout for weeks={weeks}. See streamed logs above.")

    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse JSON from child for weeks={weeks}: {e}. "
            f"See streamed logs above. Raw stdout was:\n{out[:500]}"
        )

def plot_methods_over_weeks(
    rows,
    outfile="metrics_by_weeks.png",
    show=True,
    annotate=True,
    band_half_factor=0.125  # subtle per-week shading width
):
    """
    Minimal, readable plot:
      - X = weeks, Y = metric
      - Lines = Improved Greedy, Evolutionary
      - Light gray shading between them (behind lines), width controlled by band_half_factor
      - Optional ratio/delta labels, with per-week manual placement via label_overrides:
          where: "right" | "left" | "top-right" | "top-left" | "bottom-right" | "bottom-left"
          dx: horizontal nudge in X units (e.g., 0.25 ~ a quarter of a week step)
          dy_frac: vertical nudge as a fraction of Y range (e.g., 0.02 = 2% of y-range)
    """
    import matplotlib.pyplot as plt

    weeks   = [r["weeks"]    for r in rows]
    improved= [r["improved"] for r in rows]
    evo     = [r["evo"]      for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))

    
    evo_line, = ax.plot(weeks, evo,       marker='o', linewidth=2, label="Evolutionary",   zorder=3)
    imp_line, = ax.plot(weeks, improved, marker='o', linewidth=2, label="Improved Greedy", zorder=3)

    # --- shaded gap width (narrow, subtle) ---
    if len(weeks) >= 2:
        step_min  = min(abs(b - a) for a, b in zip(weeks[:-1], weeks[1:]))
        band_half = band_half_factor * step_min
    else:
        band_half = 0.3 * band_half_factor / 0.125  # reasonable default

    # Minimal neutral gray shading behind lines
    band_color = "0.6"   # mid gray
    band_alpha = 0.12    # low opacity so it doesn't dominate

    y_all   = improved + evo
    y_min   = min(y_all)
    y_max   = max(y_all)
    y_range = max(1e-9, y_max - y_min)
    # a bit of headroom for labels
    top_pad = 0.12
    ax.set_ylim(y_min - 0.05*y_range, y_max + 0.12*y_range)

    # Draw shaded gaps
    for x, y_imp, y_evo in zip(weeks, improved, evo):
        y_low, y_high = sorted((y_imp, y_evo))
        ax.fill_between([x - band_half, x + band_half], y_low, y_high,
                        color=band_color, alpha=band_alpha, zorder=1)

    # Labels just above Evo points
    if annotate:
        UMINUS = "−"  # nicer minus
        for x, y_imp, y_evo in zip(weeks, improved, evo):
            ratio = (y_imp / y_evo * 100.0) if y_evo != 0 else float("nan")
            delta = y_imp - y_evo
            label = f"{ratio:.2f}% ({UMINUS}{abs(delta):.1f})" if delta < 0 else f"{ratio:.2f}% (+{delta:.1f})"

            # Anchor to Improved (lower) and place just above it
            x_lab = x
            y_lab = y_imp + 0.03 * y_range

            ax.text(
                x_lab, y_lab, label,
                ha="center", va="bottom", fontsize=9, color="0.15",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="0.90",   # light gray pill
                    edgecolor="0.75",
                    alpha=0.90          # 90% opaque for readability over lines
                ),
                zorder=5
            )

    ax.set_title("Scheduling Metric by the number of Weeks (Improved Greedy vs Evolutionary)")
    ax.set_xlabel("Turnament length in weeks")
    ax.set_ylabel("Metric score")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    fig.tight_layout()

    try:
        fig.savefig(outfile, dpi=200)
        print(f"\nSaved chart to: {outfile}")
    except Exception as e:
        print(f"Warning: failed to save chart: {e}")

    if show:
        try:
            plt.show()
        except Exception:
            pass

def main():
    # CHILD MODE: compute once, print clean JSON ONLY to stdout.
    if "--child" in sys.argv:
        with contextlib.redirect_stdout(sys.stderr):
            res = compute_once()
        print(json.dumps(res), flush=True)
        return

    # PARENT MODE
    if not WEEK_GRID:
        print("WEEK_GRID is empty. Edit WEEK_GRID at the top of this file.")
        sys.exit(1)

    rows = []
    for w in WEEK_GRID:
        print(f"\n=== Launching week {w} ===", flush=True)
        res = run_child_with_weeks(w)
        rows.append({
            "weeks": w,
            "random":   res["random_metric"],
            "greedy":   res["greedy_metric"],
            "improved": res["improved_greedy_metric"],
            "evo":      res["evo_best_metric"],
        })

    print("weeks\trandom\t\tgreedy\t\timproved\t\tevo")
    for r in rows:
        print(f'{r["weeks"]}\t{r["random"]:.4f}\t{r["greedy"]:.4f}\t{r["improved"]:.4f}\t{r["evo"]:.4f}')

    # Plot with non-overlapping labels
    outfile = "metrics_by_weeks.png" if len(rows) > 1 else f"metrics_week_{rows[0]['weeks']}.png"

    plot_methods_over_weeks(
        rows,
        outfile=("metrics_by_weeks.png" if len(rows) > 1 else f"metrics_week_{rows[0]['weeks']}.png"),
        show=True,
        annotate=True,
        band_half_factor=0.125
    )

if __name__ == "__main__":
    main()
