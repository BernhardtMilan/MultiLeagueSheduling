from init import *
from collections import defaultdict
import itertools

def parse_occupied_rule(rule, days_of_week, time_slots, pitches):
    """
    Parses a single occupied rule and returns a set of (week, day, time, pitch) tuples.
    """
    result = set()
    parts = rule.strip().split()
    if len(parts) < 2:
        print(f"Invalid rule skipped: {rule}")
        return result

    week = parts[1] if parts[0].lower() == "week" else None
    if not week:
        return result
    week_key = f"Week {week}"

    day = next((d for d in days_of_week if d in parts), None)
    time = next((t for t in time_slots if t in rule), None)
    pitch = None
    for i, p in enumerate(parts):
        if p.lower() == "pitch" and i + 1 < len(parts):
            try:
                pitch = int(parts[i + 1])
            except:
                pass

    if day and time and pitch:
        result.add((week_key, day, time, pitch))
    elif day and time:
        for p in pitches:
            result.add((week_key, day, time, p))
    elif day and pitch:
        for t in time_slots:
            result.add((week_key, day, t, pitch))
    elif day:
        for t in time_slots:
            for p in pitches:
                result.add((week_key, day, t, p))
    else:
        print(f"Could not interpret rule: {rule}")

    return result

def ScheduleValidator(schedule, devided_leagues, division_counts):
    pitches = [1, 2, 3, 4]

    # Parse occupied rules into a set of forbidden slots
    occupied_set = set()
    for rule in occupied_rules:
        occupied_set.update(parse_occupied_rule(rule, days_of_week, time_slots, pitches))

    # Build team → subleague map
    subleague_sets = []
    team_to_subleague = {}

    for i, league in enumerate(devided_leagues):
        teams = set(league.keys())
        subleague_sets.append(teams)
        for team in teams:
            team_to_subleague[team] = i

    # Count matches
    match_count = defaultdict(lambda: defaultdict(int))

    for week in weeks:
        for day in days_of_week:
            for time in time_slots:
                for pitch in pitches:
                    slot = schedule[week][day][time][pitch]

                    if slot == "OCCUPIED TIMESLOT":
                        if (week, day, time, pitch) not in occupied_set:
                            print(f"Invalid OCCUPIED TIMESLOT at {week} {day} {time} pitch {pitch}")
                            return False
                    elif isinstance(slot, tuple):
                        team1, team2 = slot
                        if team1 == team2:
                            print(f"A team is playing itself: {team1}")
                            return False
                        if team1 not in team_to_subleague or team2 not in team_to_subleague:
                            print(f"Unknown team in match: {team1}, {team2}")
                            return False
                        if team_to_subleague[team1] != team_to_subleague[team2]:
                            print(f"Cross-subleague match: {team1} vs {team2}")
                            return False

                        match_count[team1][team2] += 1
                        match_count[team2][team1] += 1

                    elif slot == "" or slot is None:
                        continue
                    else:
                        print(f"Invalid slot content at {week} {day} {time} pitch {pitch}: {slot}")
                        return False

    # Validate match counts
    for subleague in subleague_sets:
        for team1, team2 in itertools.combinations(subleague, 2):
            count = match_count[team1][team2]
            if count != 1:
                print(f"Invalid match count between {team1} and {team2}: {count}")
                return False

    print("Schedule is valid.")
    return True