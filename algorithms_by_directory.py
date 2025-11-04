import os
import sys
import json
import subprocess
from pathlib import Path
import contextlib
from datetime import datetime

DATASET_GRID = [
    #"fully_random",
    # "old",
    "real_world_like",
    #"optimal",
    #"real_world_like_optimal",
]

SELF = Path(__file__).resolve()

def compute_once() -> dict:
    """
    Runs one full experiment for the CURRENT process using SORSOLO_DATASET
    (and SORSOLO_WEEKS if set), returns a rich dict of metrics & breakdowns.
    """
    from init import weights
    from evolutionary import evolutionary
    from non_ai_sorts import run_non_ai_sorts
    from benchmark_adaptive_tabu_search import ATS
    from benchmark_genetic_alg import pygadBinaryEvo

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

    evo_best_metric, evo_best_draw_structure, evo_best_scores, evo_best_value_counts, evo_time_elapsed = evolutionary(
        impoved_greedy_draw_data[0],
        team_schedules,
        possible_max_metric,
        league_teams,
        plot=False
    )

    ats_best_metric, ats_best_draw_structure, ats_best_scores, ats_best_value_counts, ats_time_elapsed = ATS(
        impoved_greedy_draw_data[0],
        team_schedules,
        league_teams,
        plot=False
    )

    ga_best_metric, ga_best_draw_structure, ga_best_scores, ga_best_value_counts, ga_time_elapsed = pygadBinaryEvo(
        impoved_greedy_draw_data[0],
        team_schedules,
        league_teams,
        plot=False
    )


    return {
        # headline metrics
        "random_metric": float(random_draw_data[1]),
        "greedy_metric": float(greedy_draw_data[1]),
        "improved_greedy_metric": float(impoved_greedy_draw_data[1]),
        "evo_best_metric": float(evo_best_metric),
        "ats_best_metric": float(ats_best_metric),
        "ga_best_metric": float(ga_best_metric),

        # weighted component scores (same order as your printout)
        "random_scores_weighted":   list(get_weighted_scores(list(random_draw_data[2]))),
        "greedy_scores_weighted":   list(get_weighted_scores(list(greedy_draw_data[2]))),
        "improved_scores_weighted": list(get_weighted_scores(list(impoved_greedy_draw_data[2]))),
        "evo_scores_weighted":      list(get_weighted_scores(list(evo_best_scores))),
        "ats_scores_weighted":      get_weighted_scores(list(ats_best_scores)),
        "ga_scores_weighted":       get_weighted_scores(list(ga_best_scores)),

        # availability buckets [bad, no answer, might, good]
        "random_value_counts":   list(random_draw_data[3]),
        "greedy_value_counts":   list(greedy_draw_data[3]),
        "improved_value_counts": list(impoved_greedy_draw_data[3]),
        "evo_value_counts":      list(evo_best_value_counts),
        "ats_value_counts":      list(ats_best_value_counts),
        "ga_value_counts":       list(ga_best_value_counts),

        "evo_time_elapsed": evo_time_elapsed,
        "ats_time_elapsed": ats_time_elapsed,
        "ga_time_elapsed": ga_time_elapsed,
    }

def run_child_with_dataset(dataset_key: str) -> dict:
    """
    Spawn this same file in --child mode with SORSOLO_DATASET set and collect JSON from stdout.
    """
    import threading

    env = os.environ.copy()
    env["SORSOLO_DATASET"] = dataset_key
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.Popen(
        [sys.executable, "-u", str(SELF), "--child"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    stdout_lines: list[str] = []

    def pump_stdout():
        try:
            for line in iter(proc.stdout.readline, ''):
                stdout_lines.append(line)
        except ValueError:
            pass

    def pump_stderr():
        try:
            for line in iter(proc.stderr.readline, ''):
                sys.stdout.write(f"[{dataset_key}] {line}")
                sys.stdout.flush()
        except ValueError:
            pass

    t_out = threading.Thread(target=pump_stdout, daemon=True)
    t_err = threading.Thread(target=pump_stderr, daemon=True)
    t_out.start(); t_err.start()
    proc.wait()
    t_out.join(); t_err.join()

    if proc.stdout:
        try: proc.stdout.close()
        except: pass
    if proc.stderr:
        try: proc.stderr.close()
        except: pass

    if proc.returncode != 0:
        raise RuntimeError(f"Run failed for dataset={dataset_key} (exit {proc.returncode}).")

    out = ''.join(stdout_lines).strip()
    if not out:
        raise RuntimeError(f"Child produced empty stdout for dataset={dataset_key}.")

    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse JSON from child for dataset={dataset_key}: {e}. "
            f"Raw stdout was:\n{out[:500]}"
        )

def main():
    # CHILD MODE: compute once and print JSON only
    if "--child" in sys.argv:
        with contextlib.redirect_stdout(sys.stderr):
            res = compute_once()
        print(json.dumps(res), flush=True)
        return

    if not DATASET_GRID:
        print("DATASET_GRID is empty. Edit DATASET_GRID at the top of this file.")
        sys.exit(1)

    all_results = []
    for key in DATASET_GRID:
        print(f"\n=== Launching dataset '{key}' ===", flush=True)
        res = run_child_with_dataset(key)
        # tag result with the dataset key
        payload = dict(res)
        payload["dataset"] = key
        all_results.append(payload)

    # Build one JSON bundle
    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_by_directory.json"

    bundle = {
        "schema_version": 1,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "weeks": int(os.getenv("SORSOLO_WEEKS", "16")),
        "datasets": all_results,
        # helpful legend for the vectors
        "score_fields": [
            "availability_weighted",
            "match_bunching_penalty_weighted",
            "idle_gap_penalty_weighted",
            "spread_reward_weighted",
            "L1_pitch_penalty_weighted",
        ],
        "availability_buckets": ["bad", "no_answer", "might", "good"],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    print(f"\nSaved full JSON to: {out_path.resolve()}")

if __name__ == "__main__":
    main()