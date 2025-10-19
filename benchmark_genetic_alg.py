# file: pygad_binary_evo.py
import copy
import random
import numpy as np
import pygad
from collections import defaultdict

from init import *
from sorsolo import flatten_matches, calculate_metric, calculate_final_metric, generate_output, initial_sort, DAY_INDEX, SLOT_INDEX
from non_ai_sorts import run_non_ai_sorts

# ----------------------------
# Helpers: slot <-> (w,d,t,p)
# ----------------------------

N_WEEKS = number_of_weeks                    # from init
N_DAYS  = 5
N_SLOTS = 6
N_PITCH = 4

SLOTS_PER_WEEK = N_DAYS * N_SLOTS * N_PITCH
TOTAL_SLOTS    = N_WEEKS * SLOTS_PER_WEEK

W_IDX = {w: i for i, w in enumerate(weeks)}      # length == N_WEEKS
D_IDX = {d: i for i, d in enumerate(days_of_week)}
T_IDX = {t: i for i, t in enumerate(time_slots)}

def slot_id_from_tuple(w_key, d_key, t_key, pitch_int):
    w = W_IDX[w_key]
    d = D_IDX[d_key]
    t = T_IDX[t_key]
    p = pitch_int - 1  # pitches are 1..4 in your draw

    # ((((w)*N_DAYS + d)*N_SLOTS + t)*N_PITCH + p)
    return (((w * N_DAYS) + d) * N_SLOTS + t) * N_PITCH + p

def slot_tuple_from_id(slot_id):
    p = slot_id % N_PITCH; slot_id //= N_PITCH
    t = slot_id % N_SLOTS; slot_id //= N_SLOTS
    d = slot_id % N_DAYS;  slot_id //= N_DAYS
    w = slot_id  # 0..N_WEEKS-1

    w_key = weeks[w]
    d_key = days_of_week[d]
    t_key = time_slots[t]
    pitch_int = p + 1
    return w_key, d_key, t_key, pitch_int

def empty_like_draw(draw):
    """Deep copy structure but clear matches, keep OCCUPIED TIMESLOT strings as-is."""
    out = {}
    for w, day_map in draw.items():
        out[w] = {}
        for d, time_map in day_map.items():
            out[w][d] = {}
            for t, pitch_map in time_map.items():
                out[w][d][t] = {}
                for p, content in pitch_map.items():
                    if content == "OCCUPIED TIMESLOT":
                        out[w][d][t][p] = "OCCUPIED TIMESLOT"
                    else:
                        out[w][d][t][p] = ""  # make empty
    return out

def get_locked_slots(draw):
    locked = set()
    for w, day_map in draw.items():
        for d, time_map in day_map.items():
            for t, pitch_map in time_map.items():
                for p, content in pitch_map.items():
                    if content == "OCCUPIED TIMESLOT":
                        locked.add(slot_id_from_tuple(w, d, t, p))
    return locked

# -----------------------------------
# Encoding: build fixed match ordering
# -----------------------------------
def enumerate_matches_with_occurrence(draw):
    """
    Returns:
      match_keys: list of unique match keys (team1, team2, occ_idx) in fixed order
      match_to_slotid: dict key -> slot_id from the given draw
    """
    matches = flatten_matches(draw)  # [(week_idx, day_key, timeslot_key, pitch, (team1,team2)), ...]
    # NOTE: flatten_matches gives week_idx as 1-based; we convert back to w_key
    # We’ll rebuild w_key from weeks list.
    occ_counter = {}
    match_keys = []
    match_to_slotid = {}

    for (week_idx, d_key, t_key, pitch, (team1, team2)) in matches:
        pair = (team1, team2)
        occ = occ_counter.get(pair, 0)
        occ_counter[pair] = occ + 1
        key = (team1, team2, occ)  # occurrence-disambiguated
        match_keys.append(key)

        w_key = weeks[week_idx - 1]
        sid = slot_id_from_tuple(w_key, d_key, t_key, pitch)
        match_to_slotid[key] = sid

    return match_keys, match_to_slotid

def encode_draw_to_chromosome(draw):
    match_keys, match_to_slotid = enumerate_matches_with_occurrence(draw)
    chromosome = np.array([match_to_slotid[k] for k in match_keys], dtype=np.int32)
    return match_keys, chromosome

# -----------------------------------
# Decoding with simple feasibility repair
# -----------------------------------
def decode_chromosome_to_draw(template_draw, match_keys, chromosome, locked_slots):
    """
    Places each match key into its slot. If a slot is locked or already used, do a simple linear-probe
    to the next available allowed slot (wrap-around). This is a light repair so we always return a draw.
    """
    draw = empty_like_draw(template_draw)
    used = set(locked_slots)

    for gene_idx, key in enumerate(match_keys):
        sid = int(chromosome[gene_idx])

        original_sid = sid
        tries = 0
        while sid in used or sid in locked_slots:
            sid = (sid + 1) % TOTAL_SLOTS     # <-- was 1920
            tries += 1
            if tries > TOTAL_SLOTS + 1:       # <-- was 1921
                sid = original_sid
                break

        used.add(sid)
        w_key, d_key, t_key, pitch = slot_tuple_from_id(sid)
        team1, team2, _occ = key
        draw[w_key][d_key][t_key][pitch] = (team1, team2)

    return draw


def calculate_metric_paper_style(draw_structure, team_schedules, league_teams):
    """
    Returns:
      fitness_scalar: float in (0, 1], larger is better
      team_week_counts: dict(team -> [count per week]), to keep downstream compatibility
    Rules:
      - If either team has availability == -2 at its assigned (day,slot) -> hard += 1
      - If both teams have availability == 2 -> no availability penalty
      - Else (any of -1, 0, +1 on either side) -> soft += 1
      - Other schedule quality terms are counted as SOFT violations:
          * bunching_penalty (extra matches in a week per team)
          * idle_gap_penalty (gaps >= 2 weeks between appearances)
          * L1_pitch_penalty (L1 not on pitch 1)
    Fitness:
      fitness = 1.0 / (1.0 + hard_violations + soft_violations)
    """
    team_week_counts = defaultdict(lambda: [0] * number_of_weeks)

    matches = flatten_matches(draw_structure)
    L1_teams = set(league_teams["L1"])

    hard_violations = 0
    soft_violations = 0

    # availability-derived penalties
    soft_availability = 0
    l1_pitch_penalty = 0

    for week_idx, day_key, timeslot_key, pitch, (team1, team2) in matches:
        day_idx = DAY_INDEX[day_key]
        slot_idx = SLOT_INDEX[timeslot_key]

        a1 = team_schedules[team1][day_idx, slot_idx]
        a2 = team_schedules[team2][day_idx, slot_idx]

        # Availability to hard/soft according to your rule
        if a1 == -2 or a2 == -2:
            hard_violations += 1
        else:
            # both are perfect -> no penalty
            if not (a1 == 2 and a2 == 2):
                soft_availability += 1  # any of (-1, 0, +1) present

        # L1 pitch soft penalty
        if team1 in L1_teams and pitch != 1:
            l1_pitch_penalty += 1
        if team2 in L1_teams and pitch != 1:
            l1_pitch_penalty += 1

        # per-week load tracking
        team_week_counts[team1][week_idx - 1] += 1
        team_week_counts[team2][week_idx - 1] += 1

    # Compute bunching & idle gaps as SOFT
    bunching_penalty = 0
    idle_gap_penalty = 0

    for counts in team_week_counts.values():
        # bunching: extra matches in any single week
        bunching_penalty += sum(c - 1 for c in counts if c > 1)

        # idle gaps (>= 2 weeks between consecutive appearances)
        weeks_played = [i for i, c in enumerate(counts) if c > 0]
        if len(weeks_played) > 1:
            gaps = [weeks_played[i+1] - weeks_played[i] - 1 for i in range(len(weeks_played) - 1)]
            idle_gap_penalty += sum(1 for g in gaps if g >= 2)

    # Aggregate soft
    soft_violations = soft_availability + bunching_penalty + idle_gap_penalty + l1_pitch_penalty

    # Paper-style fitness (pygad maximizes this)
    fitness_scalar = 1.0 / (1.0 + hard_violations + soft_violations)

    return fitness_scalar, team_week_counts

def count_violations_paper_style(draw_structure, team_schedules, league_teams):
    team_week_counts = defaultdict(lambda: [0] * number_of_weeks)
    matches = flatten_matches(draw_structure)
    L1_teams = set(league_teams["L1"])

    hard = 0
    soft_av = 0
    l1_pitch = 0

    for week_idx, day_key, timeslot_key, pitch, (team1, team2) in matches:
        day_idx = DAY_INDEX[day_key]
        slot_idx = SLOT_INDEX[timeslot_key]
        a1 = team_schedules[team1][day_idx, slot_idx]
        a2 = team_schedules[team2][day_idx, slot_idx]

        if a1 == -2 or a2 == -2:
            hard += 1
        else:
            if not (a1 == 2 and a2 == 2):
                soft_av += 1

        if team1 in L1_teams and pitch != 1:
            l1_pitch += 1
        if team2 in L1_teams and pitch != 1:
            l1_pitch += 1

        team_week_counts[team1][week_idx - 1] += 1
        team_week_counts[team2][week_idx - 1] += 1

    bunching = 0
    idle_gap = 0
    for counts in team_week_counts.values():
        bunching += sum(c - 1 for c in counts if c > 1)
        weeks_played = [i for i, c in enumerate(counts) if c > 0]
        if len(weeks_played) > 1:
            gaps = [weeks_played[i+1] - weeks_played[i] - 1 for i in range(len(weeks_played) - 1)]
            idle_gap += sum(1 for g in gaps if g >= 2)

    soft_total = soft_av + bunching + idle_gap + l1_pitch
    return hard, soft_av, bunching, idle_gap, l1_pitch, soft_total

# -----------------------------------
# GA wrapper
# -----------------------------------
def pygadBinaryEvo(draw, team_schedules, league_teams, plot=True,
                   population_size=POPULATION_SIZE, generations=int(GENERATIONS/1000), mutation_prob=0.50, crossover_type="single_point"):
    """
    Match-centric chromosome:
      - len = #matches
      - gene = slot_id in [0..1919]
    """
    # Build encoding & constraints
    match_keys, base_chromosome = encode_draw_to_chromosome(draw)
    n_genes = len(base_chromosome)
    locked_slots = get_locked_slots(draw)
    allowed_slots = np.array([sid for sid in range(TOTAL_SLOTS) if sid not in locked_slots], dtype=np.int32)

    print("")
    print(f"Starting PyGAD baseline | pop={population_size} gens={generations} "
          f"mutation_prob={mutation_prob} crossover={crossover_type} genes={n_genes}")

    # Seed initial population = base + jittered copies
    def seed_population(n):
        pop = []
        for i in range(n):
            if i == 0:
                pop.append(base_chromosome.copy())
            else:
                child = base_chromosome.copy()
                # light random relocations
                k = max(1, n_genes // 50)  # ~2% of genes moved initially
                for _ in range(k):
                    gi = random.randrange(n_genes)
                    child[gi] = int(allowed_slots[random.randrange(len(allowed_slots))])
                pop.append(child)
        return np.array(pop, dtype=np.int32)

    initial_pop = seed_population(population_size)

    # Fitness (you’ll likely replace this with your tuned variant later)
    def make_fitness_func(draw, match_keys, locked_slots, team_schedules, league_teams):
        def fitness_func(ga_instance, solution, solution_idx):
            decoded = decode_chromosome_to_draw(draw, match_keys, solution, locked_slots)
            fitness_val, _ = calculate_metric_paper_style(decoded, team_schedules, league_teams)
            return float(fitness_val)  # PyGAD maximizes this
        return fitness_func

    # Optional: callback to see progress
    best_history = []
    last_report_fit = None

    def on_generation(ga_inst):
        nonlocal last_report_fit
        best_sol, best_fit, _ = ga_inst.best_solution()
        best_history.append(best_fit)

        gen = ga_inst.generations_completed  # 1-based after first generation completes
        if gen % 10 == 0:
            delta = (best_fit - last_report_fit) if last_report_fit is not None else 0.0

            # Decode best only every 10 gens for light logging
            decoded_best = decode_chromosome_to_draw(draw, match_keys, best_sol, locked_slots)
            hard, soft_av, bunching, idle_gap, l1_pitch, soft_total = count_violations_paper_style(
                decoded_best, team_schedules, league_teams
            )

            print(f"[Gen {gen:4d}] best_fitness={best_fit:.6f} "
                  f"hard={hard}  soft_total={soft_total} ")

            last_report_fit = best_fit

    # Gene space: any allowed slot; pygad keeps values within the given space for each gene.
    gene_space = [allowed_slots] * n_genes

    fitness_func = make_fitness_func(draw, match_keys, locked_slots, team_schedules, league_teams)

    ga = pygad.GA(
        initial_population=initial_pop,
        num_generations=generations,
        num_parents_mating=max(2, population_size // 2),
        fitness_func=fitness_func,             # <-- now 3-arg
        gene_space=gene_space,
        gene_type=int,
        mutation_probability=mutation_prob,
        mutation_type="random",
        crossover_type=crossover_type,
        keep_elitism=max(1, population_size // 10),
        on_generation=on_generation,           # def on_generation(ga_instance): ... (already OK)
        allow_duplicate_genes=True
    )

    ga.run()

    # --------------------------
    # Build 'draws' from final pop
    # --------------------------
    final_population = ga.population            # shape: (pop_size, n_genes)
    draws = []
    for sol in final_population:
        d = decode_chromosome_to_draw(draw, match_keys, sol, locked_slots)
        draws.append(d)

    # Let your downstream code select the very best with calculate_final_metric
    final_metrics = [(calculate_final_metric(d, team_schedules, league_teams), d) for d in draws]
    final_metrics.sort(key=lambda x: -x[0][0])  # Sort by the scalar metric

    best_metric, best_scores, _, best_value_counts = (
        final_metrics[0][0][0],
        final_metrics[0][0][1],
        final_metrics[0][0][2],
        final_metrics[0][0][3],
    )
    best_draw = final_metrics[0][1]

    # Optional: quick plot using your history (only if you want)
    if plot and len(best_history) > 1:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7,4))
            plt.plot(best_history, marker='o')
            plt.title(f"pygadBinaryEvo: best fitness over generations (pop={population_size})")
            plt.xlabel("Generation")
            plt.ylabel("Best fitness")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass
    
        print(f"Final best fitness: {best_metric}")
        print("Final best draw scores:")
        print(best_scores)
        print("Weighted scores:")
        print(f'[{weights["availability"] * best_scores[0]}, {weights["match_bunching_penalty"] * best_scores[1]}, {weights["idle_gap_penalty"] * best_scores[2]}, {weights["spread_reward"] * best_scores[3]}, {weights["L1_pitch_penalty"] * best_scores[4]}]')
        print("")
        print("Final best draw value_counts:")
        print(best_value_counts)

    return best_metric, best_draw, best_scores, best_value_counts

if __name__ == "__main__":
    #(team_schedules, possible_max_metric, league_teams, random_draw_data, greedy_draw_data, improved_greedy_draw_data) = run_non_ai_sorts()
    #draw = improved_greedy_draw_data[0]
    draw, team_schedules, possible_max_metric, league_teams = initial_sort(directory, plot=False)
    _, _, _, _ = pygadBinaryEvo(draw, team_schedules, league_teams, plot=True)