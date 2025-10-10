import json
import random
import os
from collections import defaultdict
import pickle
import numpy as np

# Time slots and days
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SLOTS = ["17:00-18:00", "18:00-19:00", "19:00-20:00", "20:00-21:00", "21:00-22:00", "22:00-23:00"]
DAY_INDEX = {day: i for i, day in enumerate(DAYS)}
SLOT_INDEX = {slot: i for i, slot in enumerate(SLOTS)}

# Desired number of teams per league
TARGET_SIZES = {
    "L1": 12,
    "L2": 24,
    "L3": 48,
    "L4": 48,
    "L5": 70
}

def reshape_to_schedule(values):
    schedule = {}
    for i, day in enumerate(DAYS):
        schedule[day] = {}
        for j, slot in enumerate(SLOTS):
            idx = i * 6 + j
            if idx < len(values):
                val = values[idx]
                schedule[day][slot] = -2 if val == 0 else val
            else:
                schedule[day][slot] = -2
    return schedule

def sum_schedule(schedule):
    return sum(
        schedule[day][slot]
        for day in schedule
        for slot in schedule[day]
    )

def convert_output_to_pickle_format(output):
    league_order = ["L1", "L2", "L3", "L4", "L5"]
    leagues = [output.get(league, {}) for league in league_order]
    team_schedules = {}

    for league in leagues:
        for team_name, schedule in league.items():
            array = np.empty((5, 6), dtype=np.int64)
            for day, times in schedule.items():
                for slot, value in times.items():
                    array[DAY_INDEX[day], SLOT_INDEX[slot]] = value
            team_schedules[team_name] = array

    return leagues, team_schedules

if __name__ == "__main__":
    with open("team_availability.json", encoding="utf-8") as f:
        raw_data = json.load(f)

    structured_data = {}
    for league, teams in raw_data.items():
        structured_data[league] = {}
        for team, values in teams.items():
            trimmed = (values + [0]*30)[:30]
            structured_data[league][team] = reshape_to_schedule(trimmed)

    # Create outputs for random and optimal selection
    random_output = {}
    optimal_output = {}

    print("\n=== League Comparison: Random vs Optimal ===\n")
    for league, teams in structured_data.items():
        team_items = list(teams.items())
        target = TARGET_SIZES.get(league)
        if not target:
            continue

        # Random selection
        random.shuffle(team_items)
        selected_random = dict(team_items[:target])

        # Optimal selection
        sorted_teams = sorted(team_items, key=lambda item: sum_schedule(item[1]), reverse=True)
        selected_optimal = dict(sorted_teams[:target])

        # Print comparison
        score_random = sum(sum_schedule(s) for s in selected_random.values())
        score_optimal = sum(sum_schedule(s) for s in selected_optimal.values())
        diff = score_optimal - score_random
        pct = (diff / score_random) * 100 if score_random else 0

        print(f"{league}:")
        print(f"  Random score:  {score_random}")
        print(f"  Optimal score: {score_optimal}")
        print(f"  Improvement:   {diff} ({pct:.2f}%)\n")

        random_output[league] = selected_random
        optimal_output[league] = selected_optimal

    print("\n=== Analytics ===")

    # 1. Per-team average scores in each league (optimal)
    print("\nAverage team score per league (optimal):")
    for league, teams in optimal_output.items():
        total_score = sum(sum_schedule(s) for s in teams.values())
        avg_score = total_score / len(teams) if teams else 0
        print(f"{league}: {avg_score:.2f} average per team")

    # 2. Per-slot total score across all optimal teams
    slot_totals = {day: defaultdict(int) for day in DAYS}

    for league_data in optimal_output.values():
        for schedule in league_data.values():
            for day in DAYS:
                for slot in SLOTS:
                    slot_totals[day][slot] += schedule[day][slot]

    print("\nTotal score per day and time slot (all optimal teams):")
    for day in DAYS:
        print(f"\n{day}:")
        for slot in SLOTS:
            print(f"  {slot}: {slot_totals[day][slot]}")

    os.makedirs("../real_world_random_tables", exist_ok=True)
    os.makedirs("../real_world_optimal_tables", exist_ok=True)

    leagues_random, schedules_random = convert_output_to_pickle_format(random_output)
    with open("../real_world_random_tables/processed_data.pkl", "wb") as f:
        pickle.dump((leagues_random, schedules_random), f)

    leagues_optimal, schedules_optimal = convert_output_to_pickle_format(optimal_output)
    with open("../real_world_optimal_tables/processed_data.pkl", "wb") as f:
        pickle.dump((leagues_optimal, schedules_optimal), f)
