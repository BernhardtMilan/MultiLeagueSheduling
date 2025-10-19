import os
import sys
import json
import subprocess
from pathlib import Path
import contextlib
from datetime import datetime

WEEK_GRID = [16]
#WEEK_GRID = [10, 11, 12, 14, 16]

SELF = Path(__file__).resolve()
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

def compute_once() -> dict:
    """
    Runs one full experiment for the CURRENT process using SORSOLO_WEEKS
    (which must already be set in the environment).
    """
    # Delayed imports so init.py reads the env var at import time
    from init import weights
    from evolutionary import evolutionary
    from non_ai_sorts import run_non_ai_sorts
    from benchmark_adaptive_tabu_search import ATS
    from benchmark_genetic_alg import pygadBinaryEvo
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

    ats_best_metric, ats_best_draw_structure, ats_best_scores, ats_best_value_counts = ATS(
        impoved_greedy_draw_data[0],
        team_schedules,
        league_teams,
        plot=False
    )

    ga_best_metric, ga_best_draw_structure, ga_best_scores, ga_best_value_counts = pygadBinaryEvo(
        impoved_greedy_draw_data[0],
        team_schedules,
        league_teams,
        plot=False
    )

    METHODS_AND_STRUCTS = [
        ("RANDOM",           random_draw_data[0]),
        ("GREEDY",           greedy_draw_data[0]),
        ("IMPROVED GREEDY",  impoved_greedy_draw_data[0]),
        ("EVOLUTIONARY",     evo_best_draw_structure),
        ("ATS",              ats_best_draw_structure),
        ("GA",               ga_best_draw_structure),
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
        "ats_best_metric": float(ats_best_metric),
        "ga_best_metric": float(ga_best_metric),

        "random_scores_weighted":   get_weighted_scores(list(random_draw_data[2])),
        "greedy_scores_weighted":   get_weighted_scores(list(greedy_draw_data[2])),
        "improved_scores_weighted": get_weighted_scores(list(impoved_greedy_draw_data[2])),
        "evo_scores_weighted":      get_weighted_scores(list(evo_best_scores)),
        "ats_scores_weighted":      get_weighted_scores(list(ats_best_scores)),
        "ga_scores_weighted":       get_weighted_scores(list(ga_best_scores)),

        "random_value_counts":   list(random_draw_data[3]),
        "greedy_value_counts":   list(greedy_draw_data[3]),
        "improved_value_counts": list(impoved_greedy_draw_data[3]),
        "evo_value_counts":      list(evo_best_value_counts),
        "ats_value_counts":      list(ats_best_value_counts),
        "ga_value_counts":       list(ga_best_value_counts),
    }

def print_human(res: dict):
    print("--------Metrics--------")
    print(f"Random: {res['random_metric']}")
    print(f"Greedy: {res['greedy_metric']}")
    print(f"Improved Greedy: {res['improved_greedy_metric']}")
    print(f"Evolutionary: {res['evo_best_metric']}")
    print(f"ATS: {res['ats_best_metric']}")
    print(f"GA: {res['ga_best_metric']}\n")
    print("Scores:")
    print("[total_availability_score, bunching_penalty, idle_gap_penalty, spread_reward, L1_pitch_penalty]")
    print(res["random_scores_weighted"])
    print(res["greedy_scores_weighted"])
    print(res["improved_scores_weighted"])
    print(res["evo_scores_weighted"])
    print(res["ats_scores_weighted"])
    print(res["ga_scores_weighted"])
    print("\nMatch availability:")
    print("[bad, no answer, might, good]")
    print(res["random_value_counts"])
    print(res["greedy_value_counts"])
    print(res["improved_value_counts"])
    print(res["evo_value_counts"])
    print(res["ats_value_counts"])
    print(res["ga_value_counts"])

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

    datasets = []
    for w in WEEK_GRID:
        print(f"\n=== Launching week {w} ===", flush=True)
        res = run_child_with_weeks(w)
        # label like metrics_by_dataset.json does ("dataset": <name>)
        labeled = dict(res)
        labeled["dataset"] = f"week_{w}"
        datasets.append(labeled)

    # Console summary (still helpful)
    print("\nweeks\trandom\t\tgreedy\t\timproved\t\tevo\t\tats\t\tga")
    for w, r in zip(WEEK_GRID, datasets):
        print(f'{w}\t{r["random_metric"]:.4f}\t{r["greedy_metric"]:.4f}\t{r["improved_greedy_metric"]:.4f}\t'
              f'{r["evo_best_metric"]:.4f}\t{r["ats_best_metric"]:.4f}\t{r["ga_best_metric"]:.4f}')

    # Build aggregate bundle (same spirit as metrics_by_dataset.json)
    bundle = {
        "schema_version": 1,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "weeks_grid": list(WEEK_GRID),     # list of weeks included
        "datasets": datasets,              # one entry per week (labeled as "dataset": "week_X")
        # helpful legends (kept the same)
        "score_fields": [
            "availability_weighted",
            "match_bunching_penalty_weighted",
            "idle_gap_penalty_weighted",
            "spread_reward_weighted",
            "L1_pitch_penalty_weighted",
        ],
        "availability_buckets": ["bad", "no_answer", "might", "good"],
    }

    out_path = RUNS_DIR / "metrics_by_weeks.json"
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved full JSON to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
