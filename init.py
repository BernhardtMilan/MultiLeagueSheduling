import enlighten

POPULATION_SIZE = 24
SURVIVORS = 3
GENERATIONS = 20000

targeting_treshold = 0.97
target_or_random = 0.7

weights = {
    "availability": 4,
    "match_bunching_penalty": -3,
    "idle_gap_penalty": -1,
    "spread_reward": 0.2
}

manager = enlighten.get_manager()
progress = manager.counter(total=GENERATIONS, desc='Evolving', unit='gen', unit_scale=True, color='bright_black')

random_directory = './fully_random_tables/'
old_directory = './old_tables/'
optimal_directory = './optimal_tables/'

directory = optimal_directory

# Allowed values in the colored schedule table
allowed_values = {-2, 0, 1, 2}

devision_strategy="pairwise" # random, knn or pairwise

weeks = [f"Week {i}" for i in range(1, 17)]
time_slots = ['17:00-18:00', '18:00-19:00', '19:00-20:00', '20:00-21:00', '21:00-22:00', '22:00-23:00']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

league_names = ["L1", "L2A", "L2B", "L3A", "L3B", "L3C", "L3D", "L4A", "L4B", "L4C", "L4D", "L5A", "L5B", "L5C", "L5D", "L5E", "L5F"]

occupied_rules = [
    "Week 14 Monday 20:00-21:00 pitch 2",
    "Week 7 Friday pitch 3",
    "Week 10 Wednesday 19:00-20:00",
    "Week 12 Tuesday"
]