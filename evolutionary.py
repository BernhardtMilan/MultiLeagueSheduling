import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter, time

from sorsolo import initial_sort, calculate_metric, calculate_final_metric, generate_output, DAY_INDEX, SLOT_INDEX
from init import *
from weights import weights

pitches = [1, 2, 3, 4]

best_metrics = []

def fancy_log(generation, metric, prev_metric, max_metric, time_elapsed):
    gain = metric - prev_metric
    percent = (metric / max_metric) * 100 if max_metric else 0

    progress.desc = f"Gen {generation:3d} | Best: {metric:.1f} ({gain:+.1f}) - {max_metric:.0f}({percent:.1f}%) | {time_elapsed:.2f}s"
    progress.update(10)
    if generation % 100 == 0:
        print(f"\nGeneration {generation:3d} | Best Metric: {metric:.1f}")

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

    return mutated_draw, (slot1, slot2, mutated_draw[w1][d1][t1][p1], mutated_draw[w2][d2][t2][p2])

def match_change_no_same_week_no_same_day(draw_structure):
    mutated_draw = draw_structure

    # Gather all valid slots
    match_slots = []
    empty_slots = []

    for week in weeks:
        for day in days_of_week:
            for time_slot in time_slots:
                for pitch in pitches:
                    content = mutated_draw[week][day][time_slot][pitch]
                    if content == "OCCUPIED TIMESLOT":
                        continue
                    slot = (week, day, time_slot, pitch)
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

    return mutated_draw, (slot1, slot2, mutated_draw[w1][d1][t1][p1], mutated_draw[w2][d2][t2][p2])

def random_until_good_match_mutation(draw_structure):

    max_attempts=20

    mutated_draw = draw_structure

    swapped_first, swapped_second = None, None

    # Find a valid match slot
    match_slot = None
    for _ in range(max_attempts):
        w = random.choice(weeks)
        d = random.choice(days_of_week)
        t = random.choice(time_slots)
        p = random.choice(pitches)

        swapped_first = mutated_draw[w][d][t][p]
        if isinstance(swapped_first, tuple):  # Found a match
            match_slot = (w, d, t, p)
            break

    if not match_slot:
        return mutated_draw  # No valid match found

    # Find a valid empty slot
    match_week, match_day, _, _ = match_slot
    swap_slot = None
    for _ in range(max_attempts):
        w = random.choice(weeks)
        d = random.choice(days_of_week)
        t = random.choice(time_slots)
        p = random.choice(pitches)

        if w == match_week or d == match_day:
            continue

        swapped_second = mutated_draw[w][d][t][p]
        if swapped_second != "OCCUPIED TIMESLOT":
            swap_slot = (w, d, t, p)
            break

    if not swap_slot:
        return mutated_draw  # No suitable target slot found

    w1, d1, t1, p1 = match_slot
    w2, d2, t2, p2 = swap_slot

    temp = mutated_draw[w1][d1][t1][p1]
    mutated_draw[w1][d1][t1][p1] = mutated_draw[w2][d2][t2][p2]
    mutated_draw[w2][d2][t2][p2] = temp

    return mutated_draw, (match_slot, swap_slot, swapped_first, swapped_second)

def calculate_availability_change(team_schedules, changes):

    change_in_availability = 0

    match_slot, swap_slot, swapped_first, swapped_second = changes

    # Extract slot info
    _ , d1, t1, _ = match_slot
    _ , d2, t2, _ = swap_slot
    day_idx_1 = DAY_INDEX[d1]
    slot_idx_1 = SLOT_INDEX[t1]
    day_idx_2 = DAY_INDEX[d2]
    slot_idx_2 = SLOT_INDEX[t2]

    # Handle the match being moved (swapped_first)
    for team in swapped_first:
        # Subtract old availability
        change_in_availability -= team_schedules[team][day_idx_1, slot_idx_1]
        # Add new availability
        change_in_availability += team_schedules[team][day_idx_2, slot_idx_2]

    # Handle second match (if there is one)
    if isinstance(swapped_second, tuple):
        for team in swapped_second:
            # Subtract old availability
            change_in_availability -= team_schedules[team][day_idx_2, slot_idx_2]
            # Add new availability
            change_in_availability += team_schedules[team][day_idx_1, slot_idx_1]
    return change_in_availability

def calculate_new_week_list(week_list, changes):
    match_slot, swap_slot, swapped_first, swapped_second = changes

    # Create deep copy of week_list to mutate safely
    new_week_list = {team: counts[:] for team, counts in week_list.items()}

    # Convert week string to index (e.g., 'Week 1' → 0)
    week_idx_1 = int(match_slot[0].split()[-1]) - 1
    week_idx_2 = int(swap_slot[0].split()[-1]) - 1

    # Handle the match that was moved *out* of week_idx_1 and *into* week_idx_2
    for team in swapped_first:
        new_week_list[team][week_idx_1] -= 1
        new_week_list[team][week_idx_2] += 1

    # If the swap involved a second match (not an empty slot), update those too
    if isinstance(swapped_second, tuple):
        for team in swapped_second:
            new_week_list[team][week_idx_2] -= 1
            new_week_list[team][week_idx_1] += 1

    return new_week_list

def caluclate_metric_change(old_metric, week_list, changes, team_schedules):

    change_in_availability = calculate_availability_change(team_schedules, changes)
    change_in_bunching_penalty = 0
    change_in_spread_reward = 0
    change_in_idle_gap_penalty = 0

    #update the week list with the changes
    new_week_list = calculate_new_week_list(week_list, changes)

    for team in new_week_list:
        new_counts = new_week_list[team]
        old_counts = week_list[team]

        # Bunching
        new_bunching = sum(c - 1 for c in new_counts if c > 1)
        old_bunching = sum(c - 1 for c in old_counts if c > 1)
        change_in_bunching_penalty += new_bunching - old_bunching

        # Spread
        new_weeks = [i for i, c in enumerate(new_counts) if c > 0]
        old_weeks = [i for i, c in enumerate(old_counts) if c > 0]
        change_in_spread_reward += len(new_weeks) - len(old_weeks)

        # Idle gaps
        if len(new_weeks) > 1:
            new_gaps = [new_weeks[i+1] - new_weeks[i] - 1 for i in range(len(new_weeks)-1)]
            new_idle = sum(1 for g in new_gaps if g >= 2)
        else:
            new_idle = 0

        if len(old_weeks) > 1:
            old_gaps = [old_weeks[i+1] - old_weeks[i] - 1 for i in range(len(old_weeks)-1)]
            old_idle = sum(1 for g in old_gaps if g >= 2)
        else:
            old_idle = 0

        change_in_idle_gap_penalty += new_idle - old_idle

    new_metric = (
        old_metric +
        weights["availability"] * change_in_availability +
        weights["match_bunching_penalty"] * change_in_bunching_penalty +
        weights["idle_gap_penalty"] * change_in_idle_gap_penalty +
        weights["spread_reward"] * change_in_spread_reward
    )

    return new_metric, new_week_list

def copy_draw_structure(draw):
    return {
        week: {
            day: {
                time: pitches.copy()  # {1: team1, 2: team2, ...}
                for time, pitches in day_data.items()
            }
            for day, day_data in week_data.items()
        }
        for week, week_data in draw.items()
    }

if __name__ == "__main__":
    draw, team_schedules, possible_max_metric = initial_sort(directory, plot=False)

    print("Initial sort done, starting evolutionary algorithm...")

    #for the metric delta we need the week counts
    # Start with 10 mutated versions of the original draw
    draws = [copy.deepcopy(draw) for _ in range(POPULATION_SIZE)]
    results = [calculate_metric(d, team_schedules) for d in draws]
    metrics = [r[0] for r in results]
    team_week_counts_list = [r[1] for r in results]

    generations_completed = 0

    patience = 2000  # How many generations to wait for improvement
    min_delta = 0  # Minimum improvement required
    stagnation_counter = 0

    mutations_per_survivor = (POPULATION_SIZE - SURVIVORS) // SURVIVORS
    extra_mutations = (POPULATION_SIZE - SURVIVORS) % SURVIVORS

    start_time_10gen = time()

    for generation in range(1, GENERATIONS + 1):
        # Evaluate all current draws
        #evaluated = []
        #for d in draws:
        #    metric = calculate_metric(d, team_schedules)
        #    evaluated.append((metric, d))

        evaluated = list(zip(metrics, draws, team_week_counts_list))
        # Sort by metric
        evaluated.sort(key=lambda x: -x[0])
        best = evaluated[:SURVIVORS]

        # Record best metric of this generation
        current_best_metric = best[0][0]

        #Early stopping logic
        if len(best_metrics) > patience:
            if current_best_metric - max(best_metrics[-patience:]) < min_delta:
                stagnation_counter += 1
                if stagnation_counter >= patience:
                    print(f"\nEarly stopping at generation {generation + 1} due to no significant improvement.")
                    break
            else:
                stagnation_counter = 0  # Reset if improvement found
        if current_best_metric == possible_max_metric:
            print(f"\nReached possible maximum metric at generation {generation + 1}")
            break
        
        best_metrics.append(current_best_metric)

        # Print best metrics of the generation
        if generation % 10 == 0:
            elapsed = time() - start_time_10gen
            start_time_10gen = time()  # Reset timer for next 10
            if len(best_metrics) > 10:
                fancy_log(generation, current_best_metric, best_metrics[-11], possible_max_metric, elapsed)
            else: # IF there is no previous generation to compare to
                progress.update(10)

        # Create next gen
        new_draws = []
        new_metrics = []
        new_team_week_counts_list = []
        for i in range(SURVIVORS):
            new_draws.append(copy_draw_structure(best[i][1]))  # survivor
            new_metrics.append(best[i][0])
            new_team_week_counts_list.append(best[i][2])

            num_mutations = mutations_per_survivor + (1 if i < extra_mutations else 0)
            for _ in range(num_mutations):
                #new_draw, changes = random_match_changes(copy_draw_structure(best[i][1]))
                #new_draw, changes = match_change_no_same_week_no_same_day(copy_draw_structure(best[i][1]))
                new_draw, changes = random_until_good_match_mutation(copy_draw_structure(best[i][1]))

                new_metric, new_team_week_counts = caluclate_metric_change(old_metric=best[i][0], week_list=best[i][2], changes=changes, team_schedules=team_schedules)

                new_draws.append(new_draw)
                new_metrics.append(new_metric)
                new_team_week_counts_list.append(new_team_week_counts)

        draws = new_draws
        metrics = new_metrics
        team_week_counts_list = new_team_week_counts_list
        generations_completed += 1

    # Get the final best draw after all generations
    final_metrics = [(calculate_final_metric(d, team_schedules), d) for d in draws]
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

    best_gen = np.argmax(best_metrics) + 1
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

