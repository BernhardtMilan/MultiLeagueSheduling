import os
import pandas as pd
import numpy as np
import random

# Define the directory where the Excel files are located
in_directory = './generalt_tablak/'
out_directory = './perfect_tables/'

random_sort = np.load('random_sort.npy', allow_pickle='TRUE').item()

def process_excel_file(excel_file):
    team_name = pd.read_excel(excel_file, header=None).iloc[0, 3].strip()

    # Hungarian to English mapping
    day_map = {
        'Hétfő': 'Monday',
        'Kedd': 'Tuesday',
        'Szerda': 'Wednesday',
        'Csütörtök': 'Thursday',
        'Péntek': 'Friday'
    }
    eng_days = list(day_map.values())
    hun_days = list(day_map.keys())
    time_slots = ['17:00-18:00', '18:00-19:00', '19:00-20:00',
                  '20:00-21:00', '21:00-22:00', '22:00-23:00']

    # Find (day, time) where this team is playing
    scheduled_slots = set()
    for week_data in random_sort.values():
        for day, day_data in week_data.items():
            for time, pitch_data in day_data.items():
                for match in pitch_data.values():
                    if isinstance(match, tuple) and team_name in match:
                        scheduled_slots.add((day, time))

    # Build a new matrix with 2s for scheduled times, else random
    new_values = []
    for time in time_slots:
        row = []
        for day in eng_days:
            if (day, time) in scheduled_slots:
                row.append(2)
            else:
                row.append(random.choice([-2, 0, 1]))
        new_values.append(row)

    # Update the original Excel file's values
    df_full = pd.read_excel(excel_file, header=None)
    for i in range(6):
        df_full.iloc[4 + i, 2:7] = new_values[i]

    output_path = os.path.join(out_directory, f"{team_name}_benchmark.xlsx")
    df_full.to_excel(output_path, index=False, header=False)
    print(f"✅ Saved benchmark for {team_name} -> {output_path}")


excel_files = [os.path.join(in_directory, f) for f in os.listdir(in_directory) if f.endswith('.xlsx')]
for excel_file in excel_files:
    process_excel_file(excel_file)