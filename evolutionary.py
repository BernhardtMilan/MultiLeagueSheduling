import random
import copy
import matplotlib.pyplot as plt
import numpy as np

from sorsolo import initial_sort, calculate_metric, generate_output
from init import *
from weights import weights

pitches = [1, 2, 3, 4]

best_metrics = []

POPULATION_SIZE = 16
SURVIVORS = 4
GENERATIONS = 20000

def fancy_log(generation, metric, prev_metric, max_metric=4400):
    progress_pct = (metric / max_metric) * 100
    gain = metric - prev_metric
    bar_length = 30
    filled_length = int(progress_pct / 100 * bar_length)
    bar = "=" * filled_length + " " * (bar_length - filled_length)

    print(f"\nGeneration {generation:3d} | Best Metric: {metric:.1f} | change in last 10 gen: {gain:+.1f}")
    #print(f"Progress: [{bar}] {progress_pct:.2f}% of max")

def random_match_changes(draw_structure):
    mutated_draw = draw_structure

    # Gather all slots that are NOT 'OCCUPIED TIMESLOT'
    swappable_slots = []
    # all slots with a match
    match_slots = []
    # all slots without a match
    empty_slots = []
    for week in weeks:
        for day in days_of_week:
            for time in time_slots:
                for pitch in pitches:
                    content = mutated_draw[week][day][time][pitch]
                    if content == "OCCUPIED TIMESLOT":
                        continue
                    slot = (week, day, time, pitch)
                    if isinstance(content, tuple):
                        match_slots.append(slot)
                    elif content == "":
                        empty_slots.append(slot)

    # Ensure we have at least 2 slots to swap and at least 1 match slot
    if not match_slots:
        return mutated_draw

    # select 1 match slot, so we cant swap two empty slots
    slot1 = random.choice(match_slots)

    # Pick a second slot (any valid non-occupied one, different from slot1)
    swappable_slots = match_slots + empty_slots
    swappable_slots.remove(slot1)

    if len(swappable_slots) < 1:
        return mutated_draw

    slot2 = random.choice(swappable_slots)

    # Swap the contents
    w1, d1, t1, p1 = slot1
    w2, d2, t2, p2 = slot2

    # Swap the contents
    temp = mutated_draw[w1][d1][t1][p1]
    mutated_draw[w1][d1][t1][p1] = mutated_draw[w2][d2][t2][p2]
    mutated_draw[w2][d2][t2][p2] = temp

    return mutated_draw

def match_change_no_same_week_no_same_day(draw_structure):
    mutated_draw = draw_structure

    # Gather all valid slots
    match_slots = []
    empty_slots = []

    for week in weeks:
        for day in days_of_week:
            for time in time_slots:
                for pitch in pitches:
                    content = mutated_draw[week][day][time][pitch]
                    if content == "OCCUPIED TIMESLOT":
                        continue
                    slot = (week, day, time, pitch)
                    if isinstance(content, tuple):
                        match_slots.append(slot)
                    elif content == "":
                        empty_slots.append(slot)

    # Ensure we have at least one match to move
    if not match_slots:
        return mutated_draw

    # Select one match slot to relocate
    slot1 = random.choice(match_slots)
    week1, day1, _, _ = slot1

    # Build swappable pool excluding same week and same day
    swappable_slots = [
        s for s in (match_slots + empty_slots)
        if s != slot1 and s[0] != week1 and s[1] != day1
    ]

    if not swappable_slots:
        return mutated_draw  # No valid second slot

    # Pick a second slot to swap with
    slot2 = random.choice(swappable_slots)

    # Swap the contents
    w1, d1, t1, p1 = slot1
    w2, d2, t2, p2 = slot2

    # Swap the contents
    temp = mutated_draw[w1][d1][t1][p1]
    mutated_draw[w1][d1][t1][p1] = mutated_draw[w2][d2][t2][p2]
    mutated_draw[w2][d2][t2][p2] = temp

    return mutated_draw

if __name__ == "__main__":
    draw, leagues, possible_max_metric = initial_sort(random_directory)
    #draw, leagues, possible_max_metric = initial_sort(prefect_directory)

    # Start with 10 mutated versions of the original draw
    draws = [random_match_changes(copy.deepcopy(draw)) for _ in range(POPULATION_SIZE)]

    generations_completed = 0

    patience = 20  # How many generations to wait for improvement
    min_delta = 0  # Minimum improvement required
    stagnation_counter = 0

    for generation in range(GENERATIONS):
        # Evaluate all current draws
        evaluated = []
        for i, d in enumerate(draws):
            metric, _, _, value_counts = calculate_metric(d, leagues)
            evaluated.append((metric, d))

            # Print only every 10th generation
            #if (generation + 1) % 10 == 0:
            #    print(f"Draw {i+1}: METRIC = {metric}")

        # Sort by metric (lower is better)
        evaluated.sort(key=lambda x: -x[0])
        best = evaluated[:SURVIVORS]

        # Record best metric of this generation
        current_best_metric = best[0][0]

        # Early stopping logic
        if len(best_metrics) > patience:
            if current_best_metric - max(best_metrics[-patience:]) < min_delta:
                stagnation_counter += 1
            else:
                stagnation_counter = 0  # Reset if improvement found

            if stagnation_counter >= patience:
                print(f"\nEarly stopping at generation {generation + 1} due to no significant improvement.")
                break
        
        best_metrics.append(current_best_metric)

        # Print best metrics of the generation
        if generation % 10 == 0:
            if len(best_metrics) > 10:
                fancy_log(generation, current_best_metric, best_metrics[-11], possible_max_metric)
            else: # IF there is no previous generation to compare to
                fancy_log(generation, current_best_metric, current_best_metric, possible_max_metric)

        # Create next gen
        new_draws = []
        for i in range(SURVIVORS):
            new_draws.append(copy.deepcopy(best[i][1]))  # survivor
            #new_draws.append(random_match_changes(copy.deepcopy(best[i][1])))  # mutated version
            new_draws.append(match_change_no_same_week_no_same_day(copy.deepcopy(best[i][1])))
            new_draws.append(match_change_no_same_week_no_same_day(copy.deepcopy(best[i][1])))

        while len(new_draws) < POPULATION_SIZE:
            seed = copy.deepcopy(random.choice(best)[1])
            #new_draws.append(random_match_changes(seed))
            new_draws.append(match_change_no_same_week_no_same_day(seed))

        draws = new_draws
        generations_completed += 1

    # Get the final best draw after all generations
    final_metrics = [(calculate_metric(d, leagues), d) for d in draws]
    final_metrics.sort(key=lambda x: -x[0][0])  # Sort by metric

    best_metric, best_scores, _, best_value_counts = final_metrics[0][0][0], final_metrics[0][0][1], final_metrics[0][0][2], final_metrics[0][0][3]

    best_draw = final_metrics[0][1]
    generate_output(best_draw, filename="best_draw_output_from_evolutionary.xlsx")

    print("\nFINAL BEST RESULT")
    print("----------------------------")
    print(f"Best METRIC: {best_metric}")
    print("[total_availability_score, bunching_penalty, idle_gap_penalty, spread_reward]")
    print("Final best draw scores:")
    print(best_scores)
    print("Weighted scores:")
    print(f'[{weights["availability"] * best_scores[0]}, {weights["match_bunching_penalty"] * best_scores[1]}, {weights["idle_gap_penalty"] * best_scores[2]}, {weights["spread_reward"] * best_scores[3]}]')
    print("")
    print("Final best draw value_counts:")
    print(best_value_counts)
    print("[bad, no answer, might, good]")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, generations_completed + 1), best_metrics, marker='o', label='Best Metric per Generation')

    plt.axhline(y=possible_max_metric, color='red', linestyle='--', label='Max Metric')

    best_gen = np.argmax(best_metrics)
    best_val = max(best_metrics)
    plt.plot(best_gen, best_val, 'ro')
    plt.text(best_gen, best_val, f"  Best: {best_val:.0f}", color='red')

    plt.title(f"Best Metric Score per Generation with {POPULATION_SIZE} individuals")
    plt.xlabel("Generation")
    plt.ylabel("Best Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

