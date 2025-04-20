import random
import copy
import matplotlib.pyplot as plt

from sorsolo import main, calculate_metric
from init import *

pitches = [1, 2, 3, 4]

best_metrics = []

POPULATION_SIZE = 8
SURVIVORS = 4
GENERATIONS = 100

def random_match_changes(draw_structure):
    mutated_draw = draw_structure

    # Gather all slots that are NOT 'OCCUPIED TIMESLOT'
    swappable_slots = []
    for week in weeks:
        for day in days_of_week:
            for time in time_slots:
                for pitch in pitches:
                    content = mutated_draw[week][day][time][pitch]
                    if content != "OCCUPIED TIMESLOT":
                        swappable_slots.append((week, day, time, pitch))

    # Ensure we have at least 2 slots to swap
    if len(swappable_slots) < 2:
        return mutated_draw

    # Randomly pick two different swappable slots
    slot1, slot2 = random.sample(swappable_slots, 2)

    w1, d1, t1, p1 = slot1
    w2, d2, t2, p2 = slot2

    # Swap the contents
    temp = mutated_draw[w1][d1][t1][p1]
    mutated_draw[w1][d1][t1][p1] = mutated_draw[w2][d2][t2][p2]
    mutated_draw[w2][d2][t2][p2] = temp

    return mutated_draw

if __name__ == "__main__":
    draw, leagues = main()

    # Start with 10 mutated versions of the original draw
    draws = [random_match_changes(copy.deepcopy(draw)) for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        # Evaluate all current draws
        evaluated = []
        for i, d in enumerate(draws):
            metric, _, value_counts = calculate_metric(d, leagues)
            evaluated.append((metric, d))

            # Print only every 10th generation
            if (generation + 1) % 10 == 0:
                print(f"Draw {i+1}: METRIC = {metric}")

        # Sort by metric (lower is better)
        evaluated.sort(key=lambda x: -x[0])
        best = evaluated[:SURVIVORS]

        # Record best metric of this generation
        best_metrics.append(best[0][0])

        # Print best metrics of the generation
        if (generation + 1) % 10 == 0:
            print(f"\n🌱 Generation {generation + 1}")
            print("-" * 30)
            print("💡 Best metrics this gen:")
            for i, (metric, _) in enumerate(best):
                print(f"  {i+1}. {metric}")
            print("")

        # Create next gen
        new_draws = []
        for i in range(SURVIVORS):
            new_draws.append(copy.deepcopy(best[i][1]))  # survivor
            new_draws.append(random_match_changes(copy.deepcopy(best[i][1])))  # mutated version

        while len(new_draws) < POPULATION_SIZE:
            seed = copy.deepcopy(random.choice(best)[1])
            new_draws.append(random_match_changes(seed))

        draws = new_draws

    # Get the final best draw after all generations
    final_metrics = [(calculate_metric(d, leagues), d) for d in draws]
    final_metrics.sort(key=lambda x: -x[0][0])  # Sort by metric

    best_metric, _, best_value_counts = final_metrics[0][0][0], final_metrics[0][0][1], final_metrics[0][0][2]

    print("\n🏁 FINAL BEST RESULT")
    print("----------------------------")
    print(f"Best METRIC: {best_metric}")
    print("Final best draw value_counts:")
    print(best_value_counts)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, GENERATIONS + 1), best_metrics, marker='o')
    plt.title("Best Metric Score per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Best Metric")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

