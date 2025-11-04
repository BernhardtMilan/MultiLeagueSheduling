import copy
import random
from time import time

from init import *
from sorsolo import calculate_metric, calculate_final_metric, initial_sort, DAY_INDEX, SLOT_INDEX
from non_ai_sorts import run_non_ai_sorts

def _get(draw, slot):
    w, d, t, p = slot
    return draw[w][d][t][p]

def _set(draw, slot, value):
    w, d, t, p = slot
    draw[w][d][t][p] = value

def _is_occupied_blocker(x):
    return x == "OCCUPIED TIMESLOT"

def _is_empty(x):
    return x == ""

def _empty_slots(draw):
    out = []
    for slot in _all_slots():
        if _is_empty(_get(draw, slot)):
            out.append(slot)
    return out

def _is_match(x):
    return isinstance(x, tuple) and len(x) == 2

def _canonical_match(m):
    return tuple(sorted(m))

def _all_slots():
    # Uses the global weeks/days/time_slots/pitches from init.py
    for w in weeks:
        for d in days_of_week:
            for t in time_slots:
                for p in [1, 2, 3, 4]:
                    yield (w, d, t, p)

def _match_slots(draw):
    # All slots that currently contain a match (not empty, not OCCUPIED)
    out = []
    for slot in _all_slots():
        x = _get(draw, slot)
        if _is_match(x):
            out.append(slot)
    return out

def _same_day_match_slots(draw, week_key, day_key):
    # All slots in (week, day, any time, any pitch) that contain a match
    out = []
    for t in time_slots:
        for p in [1, 2, 3, 4]:
            slot = (week_key, day_key, t, p)
            x = _get(draw, slot)
            if _is_match(x):
                out.append(slot)
    return out

def _same_day_empty_slots(draw, week_key, day_key):
    out = []
    for t in time_slots:
        for p in [1, 2, 3, 4]:
            slot = (week_key, day_key, t, p)
            if _is_empty(_get(draw, slot)):
                out.append(slot)
    return out

def _same_week_empty_slots(draw, week_key):
    out = []
    for d in days_of_week:
        for t in time_slots:
            for p in [1, 2, 3, 4]:
                slot = (week_key, d, t, p)
                if _is_empty(_get(draw, slot)):
                    out.append(slot)
    return out

def _non_blocked_slots(draw):
    """All slots that are not 'OCCUPIED TIMESLOT' (so empty or match)."""
    out = []
    for slot in _all_slots():
        if not _is_occupied_blocker(_get(draw, slot)):
            out.append(slot)
    return out

def copy_draw_structure(draw):
    return {
        week: {
            day: {
                time: pitches.copy()
                for time, pitches in day_data.items()
            }
            for day, day_data in week_data.items()
        }
        for week, week_data in draw.items()
    }

def penalty(draw, team_schedules, league_teams):
    """Lower is better. We use -metric to convert your 'maximize' into 'minimize'."""
    metric, _ = calculate_metric(draw, team_schedules, league_teams)
    return -metric

def move_key(move):
    """
    Tabu signature (swap-only version for now):
      - swap: forbid the SAME unordered pair of slots from being swapped again soon
    This avoids calling into the draw (keeps it robust for testing).
    """
    kind = move[0]
    if kind == "swap":
        _, a, b = move
        # canonical unordered pair of slots
        pair = tuple(sorted([a, b]))
        return ("swap", pair)

    elif kind == "relocate":
        # If/when you add relocate, consider embedding the match identity in the move.
        _, src, dst = move
        return ("relocate-slot", src, dst)

    return move  # fallback


def violates_hard(move, draw, team_schedules, league_teams):
    """
    Fast feasibility checks ONLY. Keep soft things in the metric.
    Hard checks we recommend:
      1) destination can't be 'OCCUPIED TIMESLOT'
      2) no team may have two matches in the SAME (week,day,time) across pitches
    """
    kind = move[0]

    def _timeslice_slots(week, day, time):
        # Return all pitch slots at the same (week, day, time)
        return [(week, day, time, p) for p in [1, 2, 3, 4]]

    if kind == "relocate":
        _, src, dst = move
        dst_val = _get(draw, dst)
        if _is_occupied_blocker(dst_val):
            return True

        match = _get(draw, src)
        if not _is_match(match):
            # relocating a non-match is pointless/invalid
            return True

        # Check team clash at destination timeslice
        w, d, t, _ = dst
        tA, tB = _canonical_match(match)
        for s in _timeslice_slots(w, d, t):
            if s == src:
                continue  # the match is moving out of this timeslice if same
            m2 = _get(draw, s)
            if _is_match(m2):
                u, v = _canonical_match(m2)
                if tA in (u, v) or tB in (u, v):
                    return True

    elif kind == "swap":
        _, a, b = move
        a_val = _get(draw, a)
        b_val = _get(draw, b)
        if _is_occupied_blocker(a_val) or _is_occupied_blocker(b_val):
            return True

        # If swapping introduces a clash in either timeslice, reject
        def _clash_if_put(match, dst_slot, src_slot_for_same_timeslice=None):
            if not _is_match(match):
                return False
            w, d, t, _ = dst_slot
            tA, tB = _canonical_match(match)
            for s in _timeslice_slots(w, d, t):
                # when swapping within the same timeslice, allow the counterpart slot
                if src_slot_for_same_timeslice and s == src_slot_for_same_timeslice:
                    continue
                m2 = _get(draw, s)
                if _is_match(m2):
                    u, v = _canonical_match(m2)
                    if tA in (u, v) or tB in (u, v):
                        return True
            return False

        # Check both directions
        # If a and b share SAME (week,day,time), pass the counterpart slot as exception
        same_slice = (a[0], a[1], a[2]) == (b[0], b[1], b[2])
        ex_a = b if same_slice else None
        ex_b = a if same_slice else None

        if _clash_if_put(a_val, b, src_slot_for_same_timeslice=ex_a):
            return True
        if _clash_if_put(b_val, a, src_slot_for_same_timeslice=ex_b):
            return True

    return False

def apply_move_inplace(draw, move):
    kind = move[0]
    if kind == "relocate":
        _, src, dst = move
        src_val = _get(draw, src)
        # Pure move: require dst is empty by generation & hard-checks
        _set(draw, dst, src_val)
        _set(draw, src, "")  # source becomes empty
    elif kind == "swap":
        _, a, b = move
        a_val = _get(draw, a)
        b_val = _get(draw, b)
        _set(draw, a, b_val)
        _set(draw, b, a_val)

def undo_move_inplace(draw, move, backup_payload):
    """
    Optional: only needed if you switch to "apply→score→undo".
    backup_payload should contain the original contents of the touched slots.
    For the above apply(), you'd store (src_before, dst_before) for relocate,
    and (a_before, b_before) for swap.
    """
    kind = move[0]
    if kind == "relocate":
        _, src, dst = move
        src_before, dst_before = backup_payload
        _set(draw, src, src_before)
        _set(draw, dst, dst_before)
    elif kind == "swap":
        _, a, b = move
        a_before, b_before = backup_payload
        _set(draw, a, a_before)
        _set(draw, b, b_before)


# ----------------------------
# Neighborhood generators (EMPTY SKELETONS)
# ----------------------------
def generate_neighborhood_nb1(draw, k=16, SAMPLES=32, MAX_TRIES=200):
    """
    Nb1: Relocate a match to an EMPTY slot (pure move). This avoids hidden clashes.
    """
    moves = []
    sources = _match_slots(draw)
    empties = _empty_slots(draw)
    if not sources or not empties:
        return moves

    seen = set()
    tries = 0
    while len(moves) < SAMPLES and tries < MAX_TRIES:
        tries += 1
        src = random.choice(sources)
        dst = random.choice(empties)
        if dst == src:
            continue
        key = ("relocate", src, dst)
        if key in seen:
            continue
        seen.add(key)
        moves.append(("relocate", src, dst))
    return moves

def generate_neighborhood_nb2(draw, k=16, SAMPLES=32):
    """
    Nb2: pick a random (week, day), then swap two matches within that day
    (across ANY timeslot and ANY pitch). Repeat until we have up to SAMPLES moves.
    """
    moves = []
    # If there are no matches this will stay empty; ATS loop will just skip.
    for _ in range(SAMPLES * 3):  # a few tries to find days with >=2 matches
        w = random.choice(weeks)
        d = random.choice(days_of_week)
        day_slots = _same_day_match_slots(draw, w, d)
        if len(day_slots) < 2:
            continue

        a = random.choice(day_slots)
        b = random.choice(day_slots)
        if a == b:
            continue

        # We don’t need to deduplicate much here; ATS caps candidates per iter anyway.
        moves.append(("swap", a, b))
        if len(moves) >= SAMPLES:
            break

    return moves

def generate_neighborhood_nb3(draw, k=16, SAMPLES=32):
    """
    Nb3: pick two arbitrary slots anywhere (that both contain a match) and swap them.
    We generate up to SAMPLES random distinct swaps.
    """
    moves = []
    slots = _match_slots(draw)
    n = len(slots)
    if n < 2:
        return moves

    seen = set()
    # Produce up to SAMPLES unique unordered pairs
    for _ in range(SAMPLES * 2):  # oversample a bit to handle duplicates
        a = random.choice(slots)
        b = random.choice(slots)
        if a == b:
            continue
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        moves.append(("swap", a, b))
        if len(moves) >= SAMPLES:
            break
    return moves

def generate_neighborhood_nb4(draw, k=16, SAMPLES=32, MAX_TRIES=300):
    """
    Nb4: Relocate a match with a *bias*:
         1) try same (week, day) → empty slot (any timeslot, any pitch)
         2) if none exist, try same week → empty slot (any day, any timeslot, any pitch)

    This keeps the move 'local' most of the time, which often reduces conflicts
    while still exploring enough when a day is full.
    """
    moves = []
    sources = _match_slots(draw)
    if not sources:
        return moves

    seen = set()
    tries = 0
    while len(moves) < SAMPLES and tries < MAX_TRIES:
        tries += 1
        src = random.choice(sources)
        w, d, t, p = src  # unpack for locality targets

        # 1) same-day empty targets first
        day_empties = _same_day_empty_slots(draw, w, d)
        if src in day_empties:  # src is a match, but guard anyway
            try:
                day_empties.remove(src)
            except ValueError:
                pass

        dst = None
        if day_empties:
            dst = random.choice(day_empties)
        else:
            # 2) fallback: same-week empties (any day)
            week_empties = _same_week_empty_slots(draw, w)
            # remove src if present (it shouldn't be empty, but safety)
            if src in week_empties:
                try:
                    week_empties.remove(src)
                except ValueError:
                    pass
            if week_empties:
                dst = random.choice(week_empties)

        if dst is None or dst == src:
            continue

        key = ("relocate", src, dst)
        if key in seen:
            continue
        seen.add(key)

        # Pure relocate (dst is guaranteed empty by construction)
        moves.append(("relocate", src, dst))

    return moves

def ATS(draw, team_schedules, league_teams, plot=True,
        MAX_ITERS=GENERATIONS,
        CANDIDATES_PER_ITER=POPULATION_SIZE,
        INIT_TABU_TENURE=20,
        MIN_TABU_TENURE=2,
        NO_IMPROVE_LIMIT=2000):
    """
    Adaptive Tabu Search loop, with aspiration and tenure adaptation.
    Neighborhoods are stubs above for you to implement.
    """

    print("")
    print(f"Starting ATS baseline | pop={POPULATION_SIZE} gens={GENERATIONS} ")

    # Ensure we start from a feasible timetable (your initial_sort should give that).
    schedule = copy_draw_structure(draw)
    best = copy_draw_structure(draw)

    best_pen = penalty(schedule, team_schedules, league_teams)
    current_pen = best_pen

    tabu_tenure = INIT_TABU_TENURE
    tabu = {}            # move_key -> expire_iter
    iters_since_improve = 0
    stagnation_counter = 0
    best_metrics = []

    start_time = time()

    for it in range(MAX_ITERS):
        # ----- Neighborhood: mix of Nb1..Nb4 -----
        # NOTE: these return empty lists until you implement them.
        nb_moves = []
        nb_moves.extend(generate_neighborhood_nb1(schedule))
        nb_moves.extend(generate_neighborhood_nb2(schedule))
        nb_moves.extend(generate_neighborhood_nb3(schedule))
        nb_moves.extend(generate_neighborhood_nb4(schedule))

        # If you want to cap work per iter:
        if len(nb_moves) > CANDIDATES_PER_ITER:
            nb_moves = random.sample(nb_moves, CANDIDATES_PER_ITER)

        # ----- Score candidates & select best admissible -----
        best_move = None
        best_move_pen = None

        for mv in nb_moves:
            if violates_hard(mv, schedule, team_schedules, league_teams):
                continue

            # Apply (temporarily), score, then rollback (or keep a copy)
            tmp = copy_draw_structure(schedule)
            apply_move_inplace(tmp, mv)
            cand_pen = penalty(tmp, team_schedules, league_teams)

            key = move_key(mv)
            tabu_active = (key in tabu and tabu[key] > it)

            # Aspiration: allow tabu if it beats global best
            if tabu_active and cand_pen >= best_pen:
                continue

            if (best_move is None) or (cand_pen < best_move_pen):
                best_move = mv
                best_move_pen = cand_pen

        if best_move is None:
            # No admissible moves (likely because neighborhoods are empty).
            # Break early for the skeleton.
            break

        # ----- Apply chosen move -----
        apply_move_inplace(schedule, best_move)
        current_pen = best_move_pen

        # ----- Update tabu -----
        key = move_key(best_move)
        tabu[key] = it + tabu_tenure

        # Expire old tabu entries
        for k in list(tabu.keys()):
            if tabu[k] <= it:
                del tabu[k]

        # ----- Best tracking & adaptation -----
        if current_pen < best_pen:
            best_pen = current_pen
            best = copy_draw_structure(schedule)
            iters_since_improve = 0
        else:
            iters_since_improve += 1

        # Adaptive tabu tenure (paper’s idea: reduce if stagnating)
        if iters_since_improve >= NO_IMPROVE_LIMIT and tabu_tenure > MIN_TABU_TENURE:
            tabu_tenure = max(MIN_TABU_TENURE, tabu_tenure - 2)
            iters_since_improve = 0
        
        current_best_metric = -best_pen  # convert to "higher is better"
        if len(best_metrics) >= patience:
            window_best = max(best_metrics[-patience:])
            if (current_best_metric - window_best) < min_delta:
                stagnation_counter += 1
                if stagnation_counter >= patience:
                    # keep history consistent, then break
                    best_metrics.append(current_best_metric)
                    print(f"\n[ATS] Early-stoping on iter {it+1} no significant improvement in {patience} generations.")
                    break
            else:
                stagnation_counter = 0
        
        best_metrics.append(current_best_metric)

        if (it + 1) % 100 == 0:
            elapsed = time() - start_time
            print(f"[ATS] iter {it+1:6d} | best_metric={current_best_metric:.4f} | tabu={tabu_tenure} | {elapsed:.1f}s")

    # ----- Final scoring using your existing function -----
    final_metric, scores, weighted, value_counts = calculate_final_metric(best, team_schedules, league_teams)
    best_metric = final_metric
    best_draw = best
    best_scores = scores
    best_value_counts = value_counts

    print("\nATS FINAL BEST RESULT")
    print("----------------------------")
    print(f"Best METRIC: {best_metric}")
    print("[availability, bunching_penalty, idle_gap_penalty, spread_reward, L1_pitch_penalty]")
    print("Scores:", best_scores)
    print("Weighted:", [weights["availability"]*best_scores[0],
                        weights["match_bunching_penalty"]*best_scores[1],
                        weights["idle_gap_penalty"]*best_scores[2],
                        weights["spread_reward"]*best_scores[3],
                        weights["L1_pitch_penalty"]*best_scores[4]])
    print("Value counts:", best_value_counts)

    time_elapsed = time() - start_time
    print(f"\nTotal time: {time_elapsed:.2f}s")

    return best_metric, best_draw, best_scores, best_value_counts, time_elapsed

if __name__ == "__main__":
    #(team_schedules, possible_max_metric, league_teams, random_draw_data, greedy_draw_data, improved_greedy_draw_data) = run_non_ai_sorts()
    #draw = improved_greedy_draw_data[0]
    draw, team_schedules, possible_max_metric, league_teams = initial_sort(directory, plot=False)
    _, _, _, _, _ = ATS(draw, team_schedules, league_teams, plot=True)