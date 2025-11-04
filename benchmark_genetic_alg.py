import random
import numpy as np
import pygad
from time import time

from init import *
from sorsolo import flatten_matches, calculate_metric, calculate_final_metric, initial_sort, WEEK_INDEX, DAY_INDEX, SLOT_INDEX
from non_ai_sorts import run_non_ai_sorts

SLOTS_PER_WEEK = 5 * 6 * 4
TOTAL_SLOTS = number_of_weeks * SLOTS_PER_WEEK

def slot_id_from_tuple(w_key, d_key, t_key, pitch_int):
    """
    Convert (week, day, time, pitch) -> flattened slot_id.
    Uses global DAY_INDEX and SLOT_INDEX from sorsolo.py.
    """
    w = WEEK_INDEX[w_key]
    d = DAY_INDEX[d_key]
    t = SLOT_INDEX[t_key]
    p = pitch_int - 1  # pitches are 1..4
    return (((w * 5) + d) * 6 + t) * 4 + p

def slot_tuple_from_id(slot_id):
    p = slot_id % 4
    slot_id //= 4
    t = slot_id % 6
    slot_id //= 6
    d = slot_id % 5
    slot_id //= 5
    w = slot_id  # 0..number_of_weeks-1

    w_key = weeks[w]
    d_key = days_of_week[d]
    t_key = time_slots[t]
    pitch_int = p + 1
    return w_key, d_key, t_key, pitch_int

def empty_like_draw(draw):
    out = {}
    for w, day_map in draw.items():
        out[w] = {}
        for d, time_map in day_map.items():
            out[w][d] = {}
            for t, pitch_map in time_map.items():
                out[w][d][t] = {}
                for p, content in pitch_map.items():
                    out[w][d][t][p] = "OCCUPIED TIMESLOT" if content == "OCCUPIED TIMESLOT" else ""
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

# Encoding:
def enumerate_matches_with_occurrence(draw):
    matches = flatten_matches(draw)  # [(week_idx, day_key, timeslot_key, pitch, (team1,team2)), ...]
    occ_counter = {}
    match_keys = []
    match_to_slotid = {}

    for (week_idx, d_key, t_key, pitch, (team1, team2)) in matches:
        pair = (team1, team2)
        occ = occ_counter.get(pair, 0)
        occ_counter[pair] = occ + 1
        key = (team1, team2, occ)
        match_keys.append(key)

        w_key = weeks[week_idx - 1]
        sid = slot_id_from_tuple(w_key, d_key, t_key, pitch)
        match_to_slotid[key] = sid

    return match_keys, match_to_slotid

def encode_draw_to_chromosome(draw):
    match_keys, match_to_slotid = enumerate_matches_with_occurrence(draw)
    chromosome = np.array([match_to_slotid[k] for k in match_keys], dtype=np.int32)
    return match_keys, chromosome

# Decoding
def decode_chromosome_to_draw(template_draw, match_keys, chromosome, locked_slots):
    draw = empty_like_draw(template_draw)
    used = set(locked_slots)

    for gene_idx, key in enumerate(match_keys):
        sid = int(chromosome[gene_idx])
        original_sid = sid
        tries = 0
        while sid in used or sid in locked_slots:
            sid = (sid + 1) % TOTAL_SLOTS
            tries += 1
            if tries > TOTAL_SLOTS + 1:
                sid = original_sid
                break

        used.add(sid)
        w_key, d_key, t_key, pitch = slot_tuple_from_id(sid)
        team1, team2, _occ = key
        draw[w_key][d_key][t_key][pitch] = (team1, team2)

    return draw

def _precompute_slot_meta():
    """Precompute (week, day_idx, slot_idx, pitch_int) for every slot_id."""
    arr = np.arange(TOTAL_SLOTS, dtype=np.int32)
    w = arr // SLOTS_PER_WEEK
    r = arr % SLOTS_PER_WEEK
    d = r // (6 * 4)
    r2 = r % (6 * 4)
    t = r2 // 4
    p = (r2 % 4) + 1  # pitch 1..4
    return w.astype(np.int16), d.astype(np.int8), t.astype(np.int8), p.astype(np.int8)

SLOT_WEEK, SLOT_DAY_IDX, SLOT_SLOT_IDX, SLOT_PITCH = _precompute_slot_meta()

def _count_violations_on_chromosome(solution, teams1_idx, teams2_idx, L1_mask, avail_by_team_day_slot):
    """
    Returns: hard, soft_av, bunching, idle_gap, l1_pitch, soft_total
    """
    w = SLOT_WEEK[solution]
    d = SLOT_DAY_IDX[solution]
    s = SLOT_SLOT_IDX[solution]
    p = SLOT_PITCH[solution]

    a1 = avail_by_team_day_slot[teams1_idx, d, s]
    a2 = avail_by_team_day_slot[teams2_idx, d, s]

    hard_mask = (a1 == -2) | (a2 == -2)
    hard = int(np.sum(hard_mask))
    perfect = (a1 == 2) & (a2 == 2)
    soft_av = int(np.sum(~hard_mask & ~perfect))

    l1_any = L1_mask[teams1_idx] | L1_mask[teams2_idx]
    l1_pitch = int(np.sum(l1_any & (p != 1)))

    T = L1_mask.shape[0]
    week_counts = np.zeros((T, number_of_weeks), dtype=np.int16)
    np.add.at(week_counts, (teams1_idx, w), 1)
    np.add.at(week_counts, (teams2_idx, w), 1)

    over1 = week_counts - 1
    over1[over1 < 0] = 0
    bunching = int(over1.sum())

    idle_gap = 0
    for t in range(T):
        wp = np.flatnonzero(week_counts[t] > 0)
        if wp.size > 1:
            gaps = np.diff(wp) - 1
            idle_gap += int(np.sum(gaps >= 2))

    soft_total = soft_av + bunching + idle_gap + l1_pitch
    return hard, soft_total


def _make_fitness_on_chromosome(teams1_idx, teams2_idx, L1_mask, avail_by_team_day_slot, hard_weight):
    def fitness_func(ga_instance, solution, solution_idx):
        hard, soft_total = _count_violations_on_chromosome(solution, teams1_idx, teams2_idx, L1_mask, avail_by_team_day_slot)
        total = hard_weight * hard + soft_total
        return float(1.0 / (1.0 + total))
    return fitness_func


def _make_move_or_swap_mutation(allowed_slots):
    allowed = np.asarray(allowed_slots, dtype=np.int32)

    def mutation(offspring, ga_instance):
        mutp = getattr(ga_instance, "mutation_probability", 0.5)
        for k in range(offspring.shape[0]):
            if random.random() >= mutp:
                continue
            child = offspring[k]
            used = np.zeros(TOTAL_SLOTS, dtype=np.bool_)
            used[child] = True
            free_mask = ~used[allowed]
            if free_mask.any():
                gi = np.random.randint(child.size)
                new_sid = allowed[np.flatnonzero(free_mask)[np.random.randint(free_mask.sum())]]
                child[gi] = new_sid
            else:
                i = np.random.randint(child.size)
                j = (i + 1 + np.random.randint(child.size - 1)) % child.size
                child[i], child[j] = child[j], child[i]
        return offspring
    return mutation


def on_generation_callback(ga_instance, start_time, teams1_idx, teams2_idx, L1_mask, avail_by_team_day_slot, best_history):
    best_sol, best_fit, _ = ga_instance.best_solution()
    best_history.append(best_fit)

    gen = ga_instance.generations_completed
    if gen % 50 == 0:
        hard, soft_total = _count_violations_on_chromosome(best_sol, teams1_idx, teams2_idx, L1_mask, avail_by_team_day_slot)
        elapsed = time() - start_time
        print(f"[Gen {gen:4d}] best_fit={best_fit:.6f} hard={hard} soft_total={soft_total} {elapsed:.2f}s")


# -----------------------------------
# GA wrapper
# -----------------------------------
def pygadBinaryEvo(draw, team_schedules, league_teams, plot=True, generations=int(GENERATIONS / 2), mutation_prob=0.95, hard_weight=10.0):
    # Encode once
    match_keys, base_chromosome = encode_draw_to_chromosome(draw)
    n_genes = len(base_chromosome)
    locked_slots = get_locked_slots(draw)
    allowed_slots = np.array([sid for sid in range(TOTAL_SLOTS) if sid not in locked_slots], dtype=np.int32)
    gene_space = [allowed_slots] * n_genes

    teams_in_matches = set([k[0] for k in match_keys]) | set([k[1] for k in match_keys])
    team_ids = {t: i for i, t in enumerate(sorted(teams_in_matches))}
    T = len(team_ids)

    teams1_idx = np.array([team_ids[k[0]] for k in match_keys], dtype=np.int32)
    teams2_idx = np.array([team_ids[k[1]] for k in match_keys], dtype=np.int32)

    L1_mask = np.zeros(T, dtype=np.bool_)
    for t in league_teams.get("L1", []):
        if t in team_ids:
            L1_mask[team_ids[t]] = True

    N_DAYS = len(days_of_week)
    N_SLOTS = len(time_slots)
    avail_by_team_day_slot = np.zeros((T, N_DAYS, N_SLOTS), dtype=np.int8)
    for name, idx in team_ids.items():
        avail_by_team_day_slot[idx, :, :] = team_schedules[name]

    start_time = time()
    best_history = []
    print("")
    print(f"Starting PyGAD baseline | pop={POPULATION_SIZE} gens={generations} mutation_prob={mutation_prob} genes={n_genes}")

    initial_pop = np.tile(base_chromosome, (POPULATION_SIZE, 1)).astype(np.int32)

    fitness_func = _make_fitness_on_chromosome(teams1_idx, teams2_idx, L1_mask, avail_by_team_day_slot, hard_weight)

    mutation_func = _make_move_or_swap_mutation(allowed_slots)

    def on_generation(ga_inst):
        on_generation_callback(ga_inst, start_time, teams1_idx, teams2_idx, L1_mask, avail_by_team_day_slot, best_history)

    ga = pygad.GA(
        initial_population=initial_pop,
        num_generations=generations,
        num_parents_mating=max(1, POPULATION_SIZE // 10), # fewer parents enables stronger exploitation
        fitness_func=fitness_func,
        parent_selection_type="rws", # rulette weel selection as in the paper
        gene_space=gene_space,
        gene_type=int,
        mutation_type=mutation_func, # own mutation function
        mutation_probability=mutation_prob, # as high as possible, since no other changes are present
        crossover_type=None,
        crossover_probability=0.0, # in my case it is not usefull to use crossover
        keep_elitism=SURVIVORS,
        on_generation=on_generation,
        stop_criteria=[f"saturate_{patience}"], # early stopping
        allow_duplicate_genes=False
    )

    ga.run()

    gens_done = ga.generations_completed
    if gens_done < generations:
        print(f"[EarlyStop] GA stopped early after {gens_done} generations due to saturation ")

    # Decode final population
    final_population = ga.population
    draws = []
    for sol in final_population:
        d = decode_chromosome_to_draw(draw, match_keys, sol, locked_slots)
        draws.append(d)

    # Evaluate final population
    final_metrics = [(calculate_final_metric(d, team_schedules, league_teams), d) for d in draws]
    final_metrics.sort(key=lambda x: -x[0][0])

    best_metric, best_scores, _, best_value_counts = (
        final_metrics[0][0][0],
        final_metrics[0][0][1],
        final_metrics[0][0][2],
        final_metrics[0][0][3],
    )
    best_draw = final_metrics[0][1]

    if plot and len(best_history) > 1:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7,4))
            plt.plot(best_history, marker='o')
            plt.title(f"pygadBinaryEvo: best fitness over generations (pop={POPULATION_SIZE})")
            plt.xlabel("Generation")
            plt.ylabel("Best fitness")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception:
            pass
    
    print("\nGA FINAL BEST RESULT")
    print("----------------------------")
    print(f"Best METRIC: {best_metric}")
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

    return best_metric, best_draw, best_scores, best_value_counts, time_elapsed

if __name__ == "__main__":
    #(team_schedules, possible_max_metric,league_teams, random_draw_data, greedy_draw_data, impoved_greedy_draw_data) = run_non_ai_sorts()
    #draw = impoved_greedy_draw_data[0]
    draw, team_schedules, possible_max_metric, league_teams = initial_sort(directory, plot=False)
    _ = pygadBinaryEvo(draw, team_schedules, league_teams, plot=True)