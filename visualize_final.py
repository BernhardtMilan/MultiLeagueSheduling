import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict
from pprint import pprint
import os

from sorsolo import calculate_final_metric, get_input_data_from_saves
from init import directory

EXCEL_PATH = "best_draw_output_from_evolutionary.xlsx"

HUNGARIAN_TO_ENGLISH_DAY = {
    "Hétfő": "Monday",
    "Kedd": "Tuesday",
    "Szerda": "Wednesday",
    "Csütörtök": "Thursday",
    "Péntek": "Friday"
}

draw_structure = {}
team_schedules = {}
scheduling_logs = []

st.markdown("""
<style>
/* Make all pitch columns exactly 25% width */
.stDataFrame table {
    table-layout: fixed;
    width: 100%;
}

.stDataFrame th, .stDataFrame td {
    width: 25% !important;
    text-align: center !important;
    vertical-align: middle !important;
    white-space: normal !important;
}
</style>
""", unsafe_allow_html=True)

def load_schedule(sheet_name="Schedule"):
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name, header=None)
        print("Excel file loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"File not found: {EXCEL_PATH}")
    except Exception as e:
        print(f"Error loading Excel: {e}")

def parse_schedule(df):
    schedule = []
    row = 0

    while row + 7 < len(df):
        week_cell = df.iloc[row, 0]

        if isinstance(week_cell, str) and "hét" in week_cell:
            current_week = week_cell.strip()

            # Go through 5 weekday blocks (each 5 columns wide)
            for block in range(5):
                col_offset = 1 + block * 4
                if col_offset + 3 >= df.shape[1]:
                    continue  # prevent index out of range

                day_cell = df.iloc[row, col_offset]

                if not isinstance(day_cell, str):
                    continue

                day_hun = day_cell.strip()
                if day_hun not in HUNGARIAN_TO_ENGLISH_DAY:
                    continue

                current_day = HUNGARIAN_TO_ENGLISH_DAY[day_hun]
                field_labels = df.iloc[row + 1, col_offset:col_offset + 4].tolist()

                for t in range(6):
                    time_row = df.iloc[row + 2 + t]
                    time_slot = time_row[0]  # e.g., "17-18"

                    for offset, field in enumerate(field_labels):
                        if pd.isna(field):
                            continue

                        match_cell = time_row[col_offset + offset]
                        if pd.isna(match_cell):
                            match = None
                        else:
                            match_str = str(match_cell).strip()
                            if match_str.upper() == "X":
                                match = "OCCUPIED TIMESLOT"
                            else:
                                teams = [team.strip() for team in match_str.split("-")]
                                match = (teams[0], teams[1]) if len(teams) == 2 else (match_str, None)

                        schedule.append({
                            "week": current_week,
                            "day": current_day,
                            "time": time_slot,
                            "field": field,
                            "match": match
                        })

            row += 8
        else:
            row += 1

    return schedule

def load_leagues(sheet_name="Leagues"):
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet_name)
        leagues = {}

        for _, row in df.iterrows():
            league = str(row.iloc[0]).strip()
            team_str = str(row.iloc[1]).strip()
            if team_str:
                teams = [t.strip() for t in team_str.split(",")]
                leagues[league] = teams

        print("Leagues loaded successfully.")
        return leagues

    except FileNotFoundError:
        print(f"File not found: {EXCEL_PATH}")
    except Exception as e:
        print(f"Error loading leagues: {e}")

def build_draw_structure(parsed_schedule):
    def normalize_time_slot(ts):
        if isinstance(ts, str) and "-" in ts:
            parts = ts.split("-")
            try:
                start = int(parts[0])
                end = int(parts[1])
                return f"{start:02d}:00-{end:02d}:00"
            except ValueError:
                return ts
        return ts

    def normalize_week(w):
        if "hét" in w:
            return f"Week {w.split('.')[0]}"
        return w

    def empty_week_dict():
        return {
            day: {} for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }

    draw_structure = {}

    for entry in parsed_schedule:
        week_raw = entry["week"]
        day = entry["day"]
        time_raw = entry["time"]
        field_label = entry["field"]
        match = entry["match"]

        week = normalize_week(week_raw)
        time = normalize_time_slot(time_raw)

        if week not in draw_structure:
            draw_structure[week] = empty_week_dict()

        if time not in draw_structure[week][day]:
            draw_structure[week][day][time] = {p: "" for p in range(1, 5)}

        try:
            pitch = int(str(field_label).split(".")[0])
            if match:
                draw_structure[week][day][time][pitch] = match
        except:
            continue

    return draw_structure

def analyze_team_scheduling(team_weeks, leagues_dict):
    from collections import defaultdict

    violating_teams = {}
    logs = []

    # Reverse lookup team → league
    team_to_league = {}
    for league, teams in leagues_dict.items():
        for team in teams:
            team_to_league[team] = league

    for team, weeks_played in team_weeks.items():
        weeks_list = sorted(weeks_played)
        week_counts = defaultdict(int)
        league = team_to_league.get(team, "Unknown League")

        for w in weeks_played:
            week_counts[w] += 1

        for week in week_counts:
            if week_counts[week] > 1:
                penalty = week_counts[week] - 1
                logs.append(f"[BUNCHING] [{league}] Team '{team}' has {week_counts[week]} matches in week {week} → Penalty: {penalty}")
                violating_teams[team] = league

        gaps = [weeks_list[i+1] - weeks_list[i] - 1 for i in range(len(weeks_list)-1)]
        for i, gap in enumerate(gaps):
            if gap >= 2:
                logs.append(f"[IDLE GAP] [{league}] Team '{team}' idle for {gap} weeks between week {weeks_list[i]} and {weeks_list[i+1]}")
                violating_teams[team] = league

        if len(set(weeks_list)) < 10:
            logs.append(f"[SPREAD] [{league}] Team '{team}' played in {len(set(weeks_list))} distinct weeks.")
            violating_teams[team] = league

    return violating_teams, logs

def extract_team_weeks(draw_structure):
    team_weeks = defaultdict(list)

    for week_idx, (week_name, week) in enumerate(draw_structure.items(), start=1):
        for day_key, day in week.items():
            for timeslot_key, timeslot in day.items():
                for match in timeslot.values():
                    if match and match != "OCCUPIED TIMESLOT":
                        team1, team2 = match
                        if team1: team_weeks[team1].append(week_idx)
                        if team2: team_weeks[team2].append(week_idx)
    return team_weeks

def render_week_tables(draw_structure, highlight_teams=None):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    time_slots = ["17:00-18:00", "18:00-19:00", "19:00-20:00", "20:00-21:00", "21:00-22:00", "22:00-23:00"]
    pitches = [1, 2, 3, 4]
    ENGLISH_TO_HUNGARIAN_DAY = {v: k for k, v in HUNGARIAN_TO_ENGLISH_DAY.items()}

    weeks_sorted = sorted(draw_structure.keys(), key=lambda w: int(w.split()[-1]))

    for week in weeks_sorted:
        week_number = week.split()[-1]
        st.subheader(f"{week_number}.hét")

        for day in days:
            hungarian_day = ENGLISH_TO_HUNGARIAN_DAY.get(day, day)
            st.markdown(f"**{hungarian_day}**")
            data = []

            for time in time_slots:
                row = []
                for pitch in pitches:
                    match = draw_structure[week][day].get(time, {}).get(pitch, "")
                    if match == "":
                        cell = " "
                    elif match == "OCCUPIED TIMESLOT":
                        cell = "⛔ Foglalt időpont ⛔"
                    elif isinstance(match, tuple):
                        team1, team2 = match
                        # Filter: only show matches involving league teams
                        if highlight_teams:
                            if team1 not in highlight_teams and team2 not in highlight_teams:
                                cell = "—"  # Hidden or dimmed
                            else:
                                cell = f"{team1} - {team2}"  # Optionally bold
                        else:
                            cell = f"{team1} - {team2}"
                    else:
                        cell = str(match)
                    row.append(cell)
                data.append(row)

            pitch_names = [f"Bogdánfy {p}" for p in pitches]
            df = pd.DataFrame(data, columns=pitch_names, index=time_slots)

            df.replace("", "\u2007", inplace=True)  # '\u2007' = "figure space", keeps width but is visually empty

            st.dataframe(df, use_container_width=True)

# 1. Set page config once
st.set_page_config(layout="wide")
st.title("Lámpafényes sorsolás 2025")

# 4. Load leagues dict only once
if "leagues_dict" not in st.session_state:
    st.session_state.leagues_dict = load_leagues()

leagues_dict = st.session_state.leagues_dict

# 2. Cache data loading and computation
@st.cache_data
def load_all_data():
    leagues, team_schedules = get_input_data_from_saves(directory, plot=False)
    df = load_schedule()
    parsed = parse_schedule(df)
    draw_structure = build_draw_structure(parsed)
    team_weeks = extract_team_weeks(draw_structure)
    violating_teams, scheduling_logs = analyze_team_scheduling(team_weeks, leagues_dict)
    return draw_structure, team_weeks, violating_teams, team_schedules, scheduling_logs

# 3. Load only once and reuse
if "draw_structure" not in st.session_state:
    draw_structure, team_weeks, violating_teams, team_schedules, scheduling_logs = load_all_data()
    st.session_state.draw_structure = draw_structure
    st.session_state.team_weeks = team_weeks
    st.session_state.violating_teams = violating_teams
    st.session_state.team_schedules = team_schedules
    st.session_state.scheduling_logs = scheduling_logs
else:
    draw_structure = st.session_state.draw_structure
    team_weeks = st.session_state.team_weeks
    violating_teams = st.session_state.violating_teams
    team_schedules = st.session_state.team_schedules
    scheduling_logs = st.session_state.get("scheduling_logs", [])

# Sidebar team selector
st.sidebar.title("Szúrés csapatra")
team_league = st.sidebar.selectbox("Liga választása a csapathoz:", ["None"] + sorted(leagues_dict.keys()))
selected_team = None
if team_league != "None":
    team_options = sorted(leagues_dict[team_league])
    selected_team = st.sidebar.selectbox("Csapat választás:", team_options)

# Sidebar league selector
st.sidebar.title("Szúrés ligára")
league_names = ["Mind"] + sorted(leagues_dict.keys())
selected_league = st.sidebar.radio("Liga választás:", league_names, horizontal=True)

if selected_team:
    highlight_teams = [selected_team]  # override everything else
    st.markdown(f"### `{selected_team}` csapat sorsolása")
elif selected_league == "Mind":
    highlight_teams = None
    st.markdown("### Teljes sorsolás")
else:
    highlight_teams = leagues_dict[selected_league]
    st.markdown(f"### `{selected_league}` liga sorsolása")

# 6. Render main schedule
render_week_tables(draw_structure, highlight_teams=highlight_teams)

# 7. Optional: Display metrics below (or add checkbox toggle)
if st.sidebar.checkbox("DEV: Show Metrics and affected teams"):
    total_metric, scores, number_of_matches, value_counts = calculate_final_metric(draw_structure, team_schedules)

    st.sidebar.markdown("---")
    st.sidebar.markdown("###Metric Details")
    st.sidebar.markdown(f"**Total Score:** `{total_metric}`")

    st.sidebar.markdown(f"• Availability: <span style='color:lime;'>{scores[0]}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"• Bunching: <span style='color:red;'>{scores[1]}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"• Idle Gaps: <span style='color:red;'>{scores[2]}</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"• Spread: <span style='color:lime;'>{scores[3]}</span>", unsafe_allow_html=True)


    st.sidebar.markdown(f"**Total Matches:** `{number_of_matches}`")

    st.sidebar.markdown(f"Keys: [bad, no answer, might, good]")
    st.sidebar.markdown(f"**Buckets:** `{value_counts}`")

    st.sidebar.markdown("###Scheduling Warnings")
    for msg in scheduling_logs:
        st.sidebar.markdown(f"- {msg}")

    st.sidebar.markdown("**Violating Teams:**")
    for team, league in sorted(violating_teams.items()):
        st.sidebar.markdown(f"- `{team}` *({league})*")
