import enlighten
import os

POPULATION_SIZE = 36
SURVIVORS = 6
GENERATIONS = 60000

targeting_treshold = 0.90
target_or_random = 0.7

#early stopping
patience = 3000
min_delta = 0.2

weights = {
    "availability": 4,
    "match_bunching_penalty": -3,
    "idle_gap_penalty": -1,
    "spread_reward": 0.2,
    "L1_pitch_penalty": -1.5,
}

manager = enlighten.get_manager()
progress = manager.counter(total=GENERATIONS, desc='Evolving', unit='gen', unit_scale=True, color='bright_black')

DIR_CHOICES = {
    "fully_random":            './input_directories/fully_random_tables/',
    "old":                     './input_directories/old_tables/',
    "optimal":                 './input_directories/optimal_tables/',
    "real_world_like_optimal": './input_directories/real_world_optimal_tables/',
    "real_world_like":  './input_directories/real_world_random_tables/',
}

# Pick directory from env, default to your current choice
DATASET_KEY = os.getenv("SORSOLO_DATASET", "real_world_like")
try:
    directory = DIR_CHOICES[DATASET_KEY]
except KeyError:
    raise ValueError(
        f"SORSOLO_DATASET='{DATASET_KEY}' is invalid. "
        f"Choose one of: {', '.join(DIR_CHOICES.keys())}."
    )

devision_strategy="pairwise" # random, knn or pairwise

number_of_weeks = WEEKS = int(os.getenv("SORSOLO_WEEKS", "16"))
weeks = [f"Week {i}" for i in range(1, int(number_of_weeks) + 1)]
time_slots = ['17:00-18:00', '18:00-19:00', '19:00-20:00', '20:00-21:00', '21:00-22:00', '22:00-23:00']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

league_names = ["L1", "L2A", "L2B", "L3A", "L3B", "L3C", "L3D", "L4A", "L4B", "L4C", "L4D", "L5A", "L5B", "L5C", "L5D", "L5E", "L5F"]

occupied_rules = [
    "Week 3 Monday 20:00-21:00 pitch 2",
    "Week 7 Friday pitch 3",
    "Week 5 Wednesday 19:00-20:00",
    "Week 2 Tuesday"
]