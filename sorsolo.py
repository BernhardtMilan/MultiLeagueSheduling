import os
import pandas as pd
import numpy as np
import random
from itertools import combinations

from schedule_validator import ScheduleValidator, parse_occupied_rule #need to fix circle import add refactor somewhere else
from init import *

# Define the directory where the Excel files are located
#directory = './tablak/'

def initialize_empty_draw_structure():
    """
    Initializes the data structure for the final draw,
    4 pitches per time slot, all time slots and pitches are initialized as empty.
    """

    schedule = {}

    for week_key in weeks:
        schedule[week_key] = {}

        for day in days_of_week:
            schedule[week_key][day] = {}

            for time_slot in time_slots:
                # Initialize 4 pitches per time slot, each pitch starts empty
                schedule[week_key][day][time_slot] = {
                    1: "", 2: "", 3: "", 4: ""
                }

    return schedule

def add_occupied_times(draw_structure):
    pitches = [1, 2, 3, 4]

    for rule in occupied_rules:
        slots_to_occupy = parse_occupied_rule(rule, days_of_week, time_slots, pitches)
        for (week, day, time, pitch) in slots_to_occupy:
            if week in draw_structure and day in draw_structure[week] and time in draw_structure[week][day]:
                draw_structure[week][day][time][pitch] = "OCCUPIED TIMESLOT"
            else:
                print(f"Invalid slot location in rule: {rule}")

    return draw_structure

# Function to process a single Excel file
def process_excel_file(excel_file):
    try:
        df = pd.read_excel(excel_file, header=None)

        # Check if the league is correctly formatted and is an integer between 1 and 5
        try:
            league = int(df.iloc[0, 1])
            if league not in range(1, 6):
                print(f"Error: Invalid league {league} in file {excel_file}. Must be between 1 and 5.")
                return None, None, None
        except (ValueError, IndexError):
            print(f"Error: Invalid or missing league in file {excel_file}")
            return None, None, None

        # Check if the team name exists and is a non-empty string
        try:
            team_name = df.iloc[0, 3]
            if not isinstance(team_name, str) or not team_name.strip():
                raise ValueError
        except (ValueError, IndexError):
            print(f"Error: Invalid or missing team name in file {excel_file}")
            return None, None, None

        # Extract the table (from row 5 to row 10 and from columns 2 to 7, which are the colored cells)
        try:
            schedule_df = df.iloc[4:10, 1:7].copy()

            # Manually iterate over the DataFrame to validate the values
            for row_idx in range(len(schedule_df)):
                for col_idx in range(1, len(schedule_df.columns)):  # Start at 1 to skip the 'Time' column
                    cell_value = schedule_df.iat[row_idx, col_idx]
                    # Validate that the cell is numeric and within allowed values, or allow NaN
                    if pd.notna(cell_value) and not (isinstance(cell_value, (int, float)) and cell_value in allowed_values):
                        print(f"Error: Invalid value {cell_value} in file {excel_file} at row {row_idx+5}, column {col_idx+1}")
                        return None, None, None
        except Exception as e:
            print(f"Error: Issue reading or validating the schedule table in file {excel_file}")
            return None, None, None

        # Rename the columns based on the days of the week
        schedule_df.columns = ['Time', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        # Initialize a dictionary to store the schedule for this team
        schedule = {}

        # Populate the schedule dictionary for this team (skip the "Time" column)
        for i, day in enumerate(days_of_week, start=1):
            schedule[day] = {}
            for j in range(len(schedule_df)):
                time_slot = schedule_df.iloc[j, 0]  # The time (first column)
                availability = schedule_df.iloc[j, i]  # The availability for that day and time
                schedule[day][time_slot] = availability

        return league, team_name, schedule

    except Exception as e:
        print(f"Error: Failed to process file {excel_file}: {e}")
        return None, None, None

def get_input_data_and_sort_to_leagues(directory):
    leagues = [{}, {}, {}, {}, {}]
    # Get the full paths of all Excel files in the directory
    excel_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.xlsx')]
    # Iterate through each Excel file, process it, and assign the team to the appropriate league
    for excel_file in excel_files:
        league, team_name, schedule = process_excel_file(excel_file)

        # Only assign the team if no errors occurred and the schedule was properly processed
        if league and team_name and schedule:
            if 1<= league <= 5:
                leagues[league-1][team_name] = schedule
            else:
                print(f"Error: Unknown league {league} for team {team_name}")

    # Now you have five dictionaries for each league
    if directory != prefect_directory:
        for i in range(5):
            print(f"Teams in League {i+1}: {len(leagues[i].keys())}")
            print(leagues[i].keys())
            print("")

    return leagues

def devide_leagues(leagues):
    devided = []
    division_counts = []

    for league in leagues:
        if len(league) > 12:
            n = len(league)
            # Calculate the minimum number of groups needed to keep sizes around 12
            num_groups = (n + 11) // 12  # This ensures each group has at most 12 teams

            # Determine base group size and how many groups will have an extra team
            base_group_size = n // num_groups
            remainder = n % num_groups

            # Create the group sizes list, with `remainder` groups of size `base_group_size + 1`
            group_sizes = [base_group_size + 1] * remainder + [base_group_size] * (num_groups - remainder)
        
            # Convert league to list of items and shuffle to randomize order
            league_items = list(league.items())
            random.shuffle(league_items)  # Shuffle the list of teams

            # Split the randomized list into groups
            start = 0
            for size in group_sizes:
                # Create a new dictionary for each divided group
                group_dict = dict(league_items[start:start + size])
                devided.append(group_dict)
                start += size
            division_counts.append(num_groups)
        else:
            devided.append(league)
            division_counts.append(1)
    return devided, division_counts

def visualize_devided_leagues(devided_leagues, division_counts):
    print("--------------------------DIVIDED LEAGUES-----------------------------------")
    # Print teams in each divided league with the original league and subdivision number
    original_league_number = 1  # Tracks the original league number
    subdivision_index = 1       # Tracks the subdivision within each league

    for i in range(len(devided_leagues)):
        print(f"Teams in League {original_league_number}/{subdivision_index}: {len(devided_leagues[i].keys())}")
        print(devided_leagues[i].keys())
        print("")

        # Check if we need to move to the next original league or just increment the subdivision
        if subdivision_index < division_counts[original_league_number - 1]:
            subdivision_index += 1  # Move to the next subdivision within the current league
        else:
            original_league_number += 1  # Move to the next original league
            subdivision_index = 1        # Reset subdivision index for the new league

def create_all_pairs(leagues):
    leagues_all_games = [[] for i in range(len(leagues))]
    league_num = 0
    for league in leagues:
        team_names = list(league.keys())
    
        # Generate all unique combinations (pairings) of teams
        leagues_all_games[league_num] = list(combinations(team_names, 2))
        league_num += 1
    return leagues_all_games

def visualize_league_games(leagues_all_games, division_counts):
    # Reset original league and subdivision indices for printing the games
    original_league_number = 1
    subdivision_index = 1

    for i in range(len(leagues_all_games)):
        league_label = f"{original_league_number}/{subdivision_index}"
        if i == 0:
            print(f"{league_label}. league games:")
            print(leagues_all_games[i])
        else:
            print(f"Sample from league {league_label}:")
            print(leagues_all_games[i][2])  # Print a sample of games

        # Update league and subdivision indices similarly as above
        if subdivision_index < division_counts[original_league_number - 1]:
            subdivision_index += 1  # Move to the next subdivision within the current league
        else:
            original_league_number += 1  # Move to the next original league
            subdivision_index = 1        # Reset subdivision index for the new league

def sort_random_matches(draw_structure, leagues_all_games):
    for league in leagues_all_games:
        for game in league:
            placed = False
            
            while not placed:
                # Randomly select a week, day, and time slot
                week_key = random.choice(list(draw_structure.keys()))
                day_key = random.choice(list(draw_structure[week_key].keys()))
                time_slot_key = random.choice(list(draw_structure[week_key][day_key].keys()))
                
                # Check each pitch in the selected time slot for availability
                for pitch, match in draw_structure[week_key][day_key][time_slot_key].items():
                    if match == '':
                        # Place the game and mark it as placed
                        draw_structure[week_key][day_key][time_slot_key][pitch] = game
                        placed = True
                        break  # Stop checking pitches once we place the game
                        
    return draw_structure

def get_team_schedule(team_name, leagues):
    """
    Helper function to find the schedule for a given team from the leagues data structure.
    """
    for league in leagues:
        if team_name in league:
            return league[team_name]
    return None  # Return None if team is not found

def calculate_metric(draw_structure, leagues):
    total_metric = 0
    number_of_matches = 0
    value_counts = [0, 0, 0, 0]  # Tracks counts of -2, 0, 1, and 2 respectively
    value_map = {-2: 0, 0: 1, 1: 2, 2: 3}  # Map availability values to indices in value_counts

    for week in draw_structure.values():
        for day_key, day in week.items():
            for timeslot_key, timeslot in day.items():
                for match in timeslot.values():
                    if match != '' and match != "OCCUPIED TIMESLOT":
                        number_of_matches+=1

                        team1, team2 = match

                        # Retrieve each team's schedule
                        team1_schedule = get_team_schedule(team1, leagues)
                        team2_schedule = get_team_schedule(team2, leagues)

                        # If a team schedule is not found, skip this match
                        if team1_schedule is None or team2_schedule is None:
                            print(f"Warning: Schedule not found for {team1} or {team2}")
                            continue

                        # Retrieve both teams' availability on this day and time slot
                        try:
                            team1_availability = team1_schedule[day_key][timeslot_key]
                            team2_availability = team2_schedule[day_key][timeslot_key]
                        except KeyError:
                            # If availability data is missing for a team or timeslot, skip this match
                            print(f"Warning: Missing data for {team1} or {team2} at {day_key} {timeslot_key}")
                            continue

                        # Calculate match metric
                        match_metric = team1_availability + team2_availability
                        total_metric += match_metric

                        # Count the values encountered
                        for value in (team1_availability, team2_availability):
                            if value in value_map:
                                value_counts[value_map[value]] += 1

    return total_metric, number_of_matches, value_counts

def calculate_metric_for_perfect_sort():
    perfect_sort = np.load('random_sort.npy', allow_pickle='TRUE').item()

    perfect_leagues = get_input_data_and_sort_to_leagues(prefect_directory)

    metric, _, value_counts = calculate_metric(perfect_sort, perfect_leagues)

    print("With a perfect sort:")

    print("")
    print("METRIC:")
    print(metric)

    print("")
    print("Match avaliability:")
    print("[bad, no answer, might, good]")
    print(value_counts)

def main():
    
    draw_structure = initialize_empty_draw_structure()

    draw_structure = add_occupied_times(draw_structure)

    leagues = get_input_data_and_sort_to_leagues(random_directory)

    devided_leagues, division_counts = devide_leagues(leagues)
    visualize_devided_leagues(devided_leagues, division_counts)

    leagues_all_games = create_all_pairs(devided_leagues)
    visualize_league_games(leagues_all_games, division_counts)

    draw_structure = sort_random_matches(draw_structure, leagues_all_games)

    print("")
    print("Samples from draw structure:")
    print("Week 2, Tuesday:")
    print(draw_structure["Week 2"]["Tuesday"])
    print("")
    print("Week 4, Friday:")
    print(draw_structure["Week 4"]["Friday"])
    print("")
    print("Week 10, Monday, 18:00-19:00:")
    print(draw_structure["Week 10"]["Monday"]["18:00-19:00"])
    print("")
    print("Week 14, Monday, 20:00-21:00:")
    print(draw_structure["Week 14"]["Monday"]["20:00-21:00"])

    metric, number_of_matches, value_counts = calculate_metric(draw_structure, leagues)

    print("")
    print("METRIC:")
    print(metric)
    print("")
    print("The number of matches:")
    print(number_of_matches)
    print("All possible matches (16weeks, 5days, 6timeslots, 4pithes) - occupied times:")
    print(16*5*6*4 - 1)

    print("")
    print("Match avaliability:")
    print("[bad, no answer, might, good]")
    print(value_counts)

    print(" ")
    print("Validating schedule...")
    ScheduleValidator(draw_structure, devided_leagues, division_counts)

    sould_save_random_sort = False
    we_are_calculating_metric_for_perfect_sort = False

    if sould_save_random_sort:
        np.save('random_sort.npy', draw_structure)
    if we_are_calculating_metric_for_perfect_sort:
        calculate_metric_for_perfect_sort()
    
    return(draw_structure, leagues)

main()