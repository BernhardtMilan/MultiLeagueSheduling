random_directory = './generalt_tablak/'
prefect_directory = './perfect_tables/'

# Allowed values in the colored schedule table
allowed_values = {-2, 0, 1, 2}

weeks = [f"Week {i}" for i in range(1, 17)]
time_slots = ['17:00-18:00', '18:00-19:00', '19:00-20:00', '20:00-21:00', '21:00-22:00', '22:00-23:00']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

occupied_rules = [
    "Week 14 Monday 20:00-21:00 pitch 2",
    "Week 7 Friday pitch 3",
    "Week 10 Wednesday 19:00-20:00",
    "Week 12 Tuesday"
]