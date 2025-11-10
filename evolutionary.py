import random
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter, time

from sorsolo import initial_sort, calculate_metric, calculate_final_metric, generate_output, DAY_INDEX, SLOT_INDEX, flatten_matches, fact
from init import *

pitches = [1, 2, 3, 4]

best_metrics = []

def fancy_log(generation, metric, prev_metric, max_metric, time_elapsed, start_time):
    gain = metric - prev_metric
    percent = (metric / max_metric) * 100 if max_metric else 0

    progress.desc = f"Gen {generation:3d} | Best: {metric:.1f} ({gain:+.1f}) - {max_metric:.0f}({percent:.1f}%) | {time_elapsed:.2f}s"
    progress.update(10)
    if generation % 100 == 0:
        print(f"\n[EVO] Generation {generation:3d} | Best Metric: {metric:.1f} | {time()-start_time:.2f}s")

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
        return mutated_draw, None

    # select 1 match slot, so we cant swap two empty slots
    slot1 = random.choice(match_slots)

    # Pick a second slot (any valid non-occupied one, different from slot1)
    swappable_slots = match_slots + empty_slots
    swappable_slots.remove(slot1)

    if len(swappable_slots) < 1:
        return mutated_draw, None

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
        return mutated_draw, None

    # Select one match slot to relocate
    slot1 = random.choice(match_slots)
    week1, day1, _, _ = slot1

    # Build swappable pool excluding same week and same day
    swappable_slots = [
        s for s in (match_slots + empty_slots)
        if s != slot1 and s[0] != week1 and s[1] != day1
    ]

    if not swappable_slots:
        return mutated_draw, None  # No valid second slot

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
        return mutated_draw, None  # No valid match found

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
        return mutated_draw, None  # No suitable target slot found

    w1, d1, t1, p1 = match_slot
    w2, d2, t2, p2 = swap_slot

    temp = mutated_draw[w1][d1][t1][p1]
    mutated_draw[w1][d1][t1][p1] = mutated_draw[w2][d2][t2][p2]
    mutated_draw[w2][d2][t2][p2] = temp

    return mutated_draw, (match_slot, swap_slot, swapped_first, swapped_second)

def targeted_mutation(draw_structure, targeted_teams, week_list, league_teams):
    mutated_draw = draw_structure
    target_slots = []
    all_swappable_slots = []

    for week in weeks:
        for day in days_of_week:
            for time in time_slots:
                for pitch in pitches:
                    content = mutated_draw[week][day][time][pitch]
                    slot = (week, day, time, pitch)

                    if content == "OCCUPIED TIMESLOT":
                        continue

                    if isinstance(content, tuple):
                        team1, team2 = content
                        if team1 in targeted_teams or team2 in targeted_teams:
                            target_slots.append(slot)

    if not target_slots:
        return mutated_draw, None

    targeted_slot = random.choice(target_slots)
    w1, d1, t1, p1 = targeted_slot
    week_index_target = int(w1.split()[1]) - 1

    # Get teams in the targeted match
    targeted_match = mutated_draw[w1][d1][t1][p1]
    if not isinstance(targeted_match, tuple):
        return mutated_draw, None
    team1, team2 = targeted_match

    # if the targeted team is L1, then only swap to pitch 1
    is_L1_match = team1 in league_teams["L1"]

    for week in weeks:
        week_index = int(week.split()[1]) - 1
        if week_index == week_index_target:
            continue  # Skip same week

        for day in days_of_week:
            for time in time_slots:
                for pitch in pitches:
                    if is_L1_match and pitch != 1:
                        continue

                    content = mutated_draw[week][day][time][pitch]
                    if content == "OCCUPIED TIMESLOT":
                        continue

                    if week_list[team1][week_index] > 0 or week_list[team2][week_index] > 0:
                        continue

                    all_swappable_slots.append((week, day, time, pitch))

    if not all_swappable_slots:
        return mutated_draw, None

    swap_slot = random.choice(all_swappable_slots)
    w2, d2, t2, p2 = swap_slot

    first = mutated_draw[w1][d1][t1][p1]
    second = mutated_draw[w2][d2][t2][p2]

    mutated_draw[w1][d1][t1][p1] = second
    mutated_draw[w2][d2][t2][p2] = first

    return mutated_draw, (targeted_slot, swap_slot, first, second)

def get_teams_to_target(draw_structure, team_schedules, league_teams):
    teams_to_target = []

    matches = flatten_matches(draw_structure)
    team_week_counts = defaultdict(lambda: [0] * 16)

    L1_teams = set(league_teams["L1"])

    for week_idx, day_key, timeslot_key, pitch, (team1, team2) in matches:
        team_week_counts[team1][week_idx - 1] += 1
        team_week_counts[team2][week_idx - 1] += 1

        # l1 teams with wrong pitch
        if pitch != 1:
            if team1 in L1_teams:
                teams_to_target.append(team1)
            if team2 in L1_teams:
                teams_to_target.append(team2)

    for team, counts in team_week_counts.items():
        weeks_played = [i for i, count in enumerate(counts) if count > 0]

        # bunching
        has_bunching_penalty = any(count > 1 for count in counts)

        # idle gaps
        has_idle_gap_penalty = False
        if len(weeks_played) > 1:
            gaps = [weeks_played[i + 1] - weeks_played[i] - 1 for i in range(len(weeks_played) - 1)]
            has_idle_gap_penalty = any(gap >= 2 for gap in gaps)

        if has_bunching_penalty or has_idle_gap_penalty:
            teams_to_target.append(team)

    return teams_to_target

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

def bunching_value(counts):
        s = 0
        for k in counts:
            if k > 1:
                s += fact(k - 1)
        return s

def idle_gap_value(counts):
        weeks_played = [i for i, c in enumerate(counts) if c > 0]
        if len(weeks_played) <= 1:
            return 0
        s = 0
        for i in range(len(weeks_played) - 1):
            g = weeks_played[i+1] - weeks_played[i] - 1
            if g >= 2:
                s += fact(g - 1)
        return s 

def spread_value(counts):
        return sum(1 for c in counts if c > 0)

def caluclate_metric_change(old_metric, week_list, changes, team_schedules, league_teams):

    if changes is None:
        return old_metric, week_list

    change_in_availability = calculate_availability_change(team_schedules, changes)
    change_in_bunching_penalty = 0
    change_in_spread_reward = 0
    change_in_idle_gap_penalty = 0
    change_in_l1_pitch_penalty = 0

    #update the week list with the changes
    new_week_list = calculate_new_week_list(week_list, changes)

    for team in new_week_list:
        new_counts = new_week_list[team]
        old_counts = week_list[team]

        # Bunching
        change_in_bunching_penalty += bunching_value(new_counts) - bunching_value(old_counts)

        # Spread
        change_in_spread_reward += spread_value(new_counts) - spread_value(old_counts)

        # Idle gaps
        change_in_idle_gap_penalty += idle_gap_value(new_counts) - idle_gap_value(old_counts)

    # L1 pitches
    L1_teams = set(league_teams["L1"])
    slot1, slot2, match1, match2 = changes

    def l1_penalty(match, pitch):
        if isinstance(match, tuple):
            team1 = match[0]
            if team1 in L1_teams and pitch != 1:
                return 1
        return 0

    old_slot1_pitch = slot1[3]
    old_slot2_pitch = slot2[3]
    old_l1_penalty = l1_penalty(match1, old_slot1_pitch) + l1_penalty(match2, old_slot2_pitch)
    new_l1_penalty = l1_penalty(match1, old_slot2_pitch) + l1_penalty(match2, old_slot1_pitch)

    change_in_l1_pitch_penalty = new_l1_penalty - old_l1_penalty

    new_metric = (
        old_metric +
        weights["availability"] * change_in_availability +
        weights["match_bunching_penalty"] * change_in_bunching_penalty +
        weights["idle_gap_penalty"] * change_in_idle_gap_penalty +
        weights["spread_reward"] * change_in_spread_reward +
        weights["L1_pitch_penalty"] * change_in_l1_pitch_penalty
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

def reorder_pitches(draw_structure, league_teams):
    # Determine the first league's teams (e.g., L1)
    first_league_name = sorted(league_teams.keys())[0]
    first_league_set = set(league_teams[first_league_name])

    def is_from_first_league(match):
        if not isinstance(match, tuple):
            return False
        team1, team2 = match
        return (team1 in first_league_set) or (team2 in first_league_set)

    for week in draw_structure:
        for day in draw_structure[week]:
            for time in draw_structure[week][day]:
                pitches = draw_structure[week][day][time]

                # Skip if pitch 1 is occupied
                if pitches.get(1) == "OCCUPIED TIMESLOT":
                    continue

                # If pitch 1 already has a first-league match, do nothing
                if is_from_first_league(pitches.get(1)):
                    continue

                for p in range(2, 5):
                    if is_from_first_league(pitches.get(p)) and pitches.get(p) != "OCCUPIED TIMESLOT":
                        pitches[1], pitches[p] = pitches[p], pitches[1]
                        break

    return draw_structure

def evolutionary(draw, team_schedules, possible_max_metric, league_teams, plot):

    print("")
    print("Starting evolutionary algorithm...")

    #for the metric delta we need the week counts
    # Start with 10 mutated versions of the original draw
    draws = [copy.deepcopy(draw) for _ in range(POPULATION_SIZE)]
    results = [calculate_metric(d, team_schedules, league_teams) for d in draws]
    metrics = [r[0] for r in results]
    team_week_counts_list = [r[1] for r in results]

    generations_completed = 0
    
    stagnation_counter = 0

    # store the problematic teams for targeted mutation
    targeted_teams = []
    targeting = False
    mutation_type_counts = {"targeted": 0, "general": 0}

    mutations_per_survivor = (POPULATION_SIZE - SURVIVORS) // SURVIVORS
    extra_mutations = (POPULATION_SIZE - SURVIVORS) % SURVIVORS

    start_time_10gen = time()
    start_time = time()

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
                    best_metrics.append(current_best_metric)
                    generations_completed += 1
                    break
            else:
                stagnation_counter = 0  # Reset if improvement found
        if current_best_metric >= possible_max_metric:
            print(f"\nReached possible maximum metric at generation {generation + 1}")
            best_metrics.append(current_best_metric)
            generations_completed += 1
            break
        
        best_metrics.append(current_best_metric)

        # Print best metrics of the generation
        if generation % 10 == 0:
            elapsed = time() - start_time_10gen
            start_time_10gen = time()  # Reset timer for next 10
            if len(best_metrics) > 10:
                fancy_log(generation, current_best_metric, best_metrics[-11], possible_max_metric, elapsed, start_time)
            else: # IF there is no previous generation to compare to
                progress.update(10)

        targeting = current_best_metric > possible_max_metric * targeting_treshold

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

                if targeting and random.random() < target_or_random:
                    #Targeted mutation
                    new_draw, changes = targeted_mutation(copy_draw_structure(best[i][1]), targeted_teams, week_list=best[i][2], league_teams=league_teams)
                    mutation_type_counts["targeted"] += 1
                else:
                    #general mutation
                    #new_draw, changes = random_match_changes(copy_draw_structure(best[i][1]))
                    #new_draw, changes = match_change_no_same_week_no_same_day(copy_draw_structure(best[i][1]))
                    new_draw, changes = random_until_good_match_mutation(copy_draw_structure(best[i][1]))
                    mutation_type_counts["general"] += 1

                new_metric, new_team_week_counts = caluclate_metric_change(old_metric=best[i][0], week_list=best[i][2], changes=changes, team_schedules=team_schedules, league_teams=league_teams)

                new_draws.append(new_draw)
                new_metrics.append(new_metric)
                new_team_week_counts_list.append(new_team_week_counts)

        if targeting and random.random() < 0.1:
            targeted_teams = get_teams_to_target(draw_structure=best[0][1], team_schedules=team_schedules, league_teams=league_teams)
            #print(f"Targeting {len(targeted_teams)} teams")

        draws = new_draws
        metrics = new_metrics
        team_week_counts_list = new_team_week_counts_list
        generations_completed += 1

    # Get the final best draw after all generations
    final_metrics = [(calculate_final_metric(d, team_schedules, league_teams), d) for d in draws]
    final_metrics.sort(key=lambda x: -x[0][0])  # Sort by metric

    best_metric, best_scores, _, best_value_counts = final_metrics[0][0][0], final_metrics[0][0][1], final_metrics[0][0][2], final_metrics[0][0][3]

    best_draw = final_metrics[0][1]

    # Manually reorder the pitches so the 1st league is always on the first pitch
    #best_draw = reorder_pitches(best_draw, league_teams)

    generate_output(best_draw, league_teams, filename="best_draw_output_from_evolutionary.xlsx")

    print("\nFINAL BEST RESULT")
    print("----------------------------")
    print(f"Best METRIC: {best_metric}")
    print("[total_availability_score, bunching_penalty, idle_gap_penalty, spread_reward, L1_pitch_penalty]")
    print("Final best draw scores:")
    print(best_scores)
    print("Weighted scores:")
    print(f'[{weights["availability"] * best_scores[0]}, {weights["match_bunching_penalty"] * best_scores[1]}, {weights["idle_gap_penalty"] * best_scores[2]}, {weights["spread_reward"] * best_scores[3]}, {weights["L1_pitch_penalty"] * best_scores[4]}]')
    print("")
    print("Final best draw value_counts:")
    print(best_value_counts)
    print("[bad, no answer, might, good]")

    time_elapsed = time() - start_time
    print(f"\nTotal time: {time_elapsed:.2f}s")

    if plot:

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

    return best_metric, best_draw, best_scores, best_value_counts, time_elapsed

if __name__ == "__main__":
    draw, team_schedules, possible_max_metric, league_teams = initial_sort(directory, plot=False)
    _, _, _, _, _ = evolutionary(draw, team_schedules, possible_max_metric, league_teams, plot=True)