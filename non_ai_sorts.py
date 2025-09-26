from sorsolo import *
from league_devider import devide_leagues
import random
from collections import defaultdict

def pure_random_sort(draw_structure, leagues_all_games):

    for league in leagues_all_games:
        for game in league:
            placed = False
            
            while not placed:
                week_key = random.choice(list(draw_structure.keys()))
                day_key = random.choice(list(draw_structure[week_key].keys()))
                time_slot_key = random.choice(list(draw_structure[week_key][day_key].keys()))
                
                for pitch, match in draw_structure[week_key][day_key][time_slot_key].items():
                    if match == '':
                        draw_structure[week_key][day_key][time_slot_key][pitch] = game
                        placed = True
                        break

    return draw_structure

def greedy_sort(draw_structure, leagues_all_games, league_teams, team_schedules):

    """
    Simple greedy using availability:
      - Flatten all matches, shuffle.
      - For each (A,B), find the empty (week,day,slot,pitch) maximizing aA+aB,
        where aX = team_schedules[X][day_index][slot_index].
      - Ties break by iteration order defined by weeks, days_of_week, time_slots, then pitch order.
      - No week-load constraints here (add in improved greedy).
    """
    # Flatten & shuffle games
    all_games = []
    for league_games in leagues_all_games:
        for g in league_games:
            if isinstance(g, (list, tuple)) and len(g) >= 2:
                all_games.append((g[0], g[1]))
    random.shuffle(all_games)

    for (A, B) in all_games:
        best_slot = None
        best_score = -10

        # Pre-get day/slot schedules
        schedA = team_schedules.get(A)
        schedB = team_schedules.get(B)

        for w in weeks:
            days = draw_structure.get(w)
            if not days:
                continue
            for di, d in enumerate(days_of_week):
                slots = days.get(d)
                if not slots:
                    continue
                for si, ts in enumerate(time_slots):
                    pitches = slots.get(ts)
                    if not pitches:
                        continue

                    # availability lookup
                    aA, aB = -2, -2
                    try:
                        aA = int(schedA[di][si]) if schedA is not None else -2
                    except Exception:
                        aA = -2
                    try:
                        aB = int(schedB[di][si]) if schedB is not None else -2
                    except Exception:
                        aB = -2

                    score = aA + aB

                    for p_key, match in pitches.items():
                        if match != '':
                            continue
                        if score > best_score:
                            best_score = score
                            best_slot = (w, d, ts, p_key)

        if best_slot is None:
            #esetleg kifogyunk helyből, de az lehetetlen
            return draw_structure

        w, d, ts, p = best_slot
        draw_structure[w][d][ts][p] = (A, B)

    return draw_structure

def improved_greedy_sort(draw_structure, leagues_all_games, league_teams, team_schedules):
    """
    One-pass greedy:
      score = (aA + aB) + week_bonus(non-conflict) + pitch_bonus(L1@P1)
    - Prefers weeks where both teams are idle; if none exist, naturally falls back to conflict weeks.
    - Prefers Pitch 1 for L1-vs-L1, but will place anywhere if none free.
    - Keeps it simple: no other constraints here.

    Expects:
      weeks, days_of_week, time_slots defined in outer scope.
      draw_structure[week][day][slot][pitch] == '' means empty.
      leagues_all_games: lists of (A,B) pairs.
      team_schedules[team]: 5x6 array-like with ints (e.g., -2, 1, 2).
    """
    
    team_week_load = defaultdict(lambda: defaultdict(int))
    for w in weeks:
        days = draw_structure.get(w, {})
        for d in days_of_week:
            slots = days.get(d, {})
            for ts in time_slots:
                for _, match in slots.get(ts, {}).items():
                    if match not in ('', None) and isinstance(match, (tuple, list)) and len(match) >= 2:
                        A0, B0 = match[0], match[1]
                        team_week_load[A0][w] += 1
                        team_week_load[B0][w] += 1

    # Flatten & shuffle games
    all_games = []
    for league_games in leagues_all_games:
        for g in league_games:
            if isinstance(g, (tuple, list)) and len(g) >= 2:
                all_games.append((g[0], g[1]))
    random.shuffle(all_games)

    l1_teams = set(league_teams.get('L1', []))
    p1_keys = {1, "1"}  # support int or str pitch keys

    for (A, B) in all_games:
        schedA = team_schedules.get(A)
        schedB = team_schedules.get(B)
        is_L1_match = (A in l1_teams and B in l1_teams)

        best_slot = None
        best_score = -1e9  # safely below min possible

        for w in weeks:
            days = draw_structure.get(w)
            if not days:
                continue

            # Large bonus ensures non-conflict weeks win when available
            no_conflict = (team_week_load[A][w] == 0 and team_week_load[B][w] == 0)
            week_bonus = 1000 if no_conflict else 0

            for di, d in enumerate(days_of_week):
                slots = days.get(d)
                if not slots:
                    continue
                for si, ts in enumerate(time_slots):
                    pitches = slots.get(ts)
                    if not pitches:
                        continue

                    # Availability
                    try:
                        aA = int(schedA[di][si]) if schedA is not None else -2
                    except Exception:
                        aA = -2
                    try:
                        aB = int(schedB[di][si]) if schedB is not None else -2
                    except Exception:
                        aB = -2

                    base = aA + aB

                    for p_key, match in pitches.items():
                        if match != '':
                            continue
                        pitch_bonus = 0.5 if (is_L1_match and p_key in p1_keys) else 0.0
                        score = week_bonus + base + pitch_bonus

                        if score > best_score:
                            best_score = score
                            best_slot = (w, d, ts, p_key)

        if best_slot is None:
            return draw_structure

        w, d, ts, p = best_slot
        draw_structure[w][d][ts][p] = (A, B)
        team_week_load[A][w] += 1
        team_week_load[B][w] += 1

    return draw_structure

def visualize_metric(draw_structure, team_schedules, league_teams, devided_leagues, division_counts, method):
    metric, scores, number_of_matches, value_counts = calculate_final_metric(draw_structure, team_schedules, league_teams)

    print(f"--------------------------{method} SORT-----------------------------------")
    print("")
    print("METRIC:")
    print(metric)
    print("")
    print("Scores:")
    print("[total_availability_score, bunching_penalty, idle_gap_penalty, spread_reward, L1_pitch_penalty]")
    print(scores)
    print("Weighted scores:")
    print(f'[{weights["availability"] * scores[0]}, {weights["match_bunching_penalty"] * scores[1]}, {weights["idle_gap_penalty"] * scores[2]}, {weights["spread_reward"] * scores[3]}, {weights["L1_pitch_penalty"] * scores[4]}]')

    print("")
    print("Match avaliability:")
    print("[bad, no answer, might, good]")
    print(value_counts)

    ScheduleValidator(draw_structure, devided_leagues, division_counts)

def run_non_ai_sorts():

    plot = False

    draw_structures = []

    for i in range(3):
        draw_structures.append(initialize_empty_draw_structure())
        draw_structures[i] = add_occupied_times(draw_structures[i])

    leagues, team_schedules = get_input_data_from_saves(directory, plot)

    devided_leagues, division_counts = devide_leagues(leagues, devision_strategy, plot)

    league_teams = build_league_team_map(devided_leagues, league_names)

    leagues_all_games = create_all_pairs(devided_leagues)

    possible_max_metric = calculate_possible_max_metric(leagues_all_games)

    random_draw_structure = pure_random_sort(draw_structures[0], leagues_all_games)
    visualize_metric(random_draw_structure, team_schedules, league_teams, devided_leagues, division_counts, "RANDOM")

    greedy_draw_structure = greedy_sort(draw_structures[1], leagues_all_games, league_teams, team_schedules)
    visualize_metric(greedy_draw_structure, team_schedules, league_teams, devided_leagues, division_counts, "GREEDY")

    impoved_greedy_draw_structure = improved_greedy_sort(draw_structures[2], leagues_all_games, league_teams, team_schedules)
    visualize_metric(impoved_greedy_draw_structure, team_schedules, league_teams, devided_leagues, division_counts, "IMPROVED GREEDY")

    return (team_schedules, possible_max_metric, league_teams, random_draw_structure, greedy_draw_structure, impoved_greedy_draw_structure)

if __name__ == "__main__":
    run_non_ai_sorts()