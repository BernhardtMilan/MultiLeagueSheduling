from init import *
from evolutionary import evolutionary
from non_ai_sorts import run_non_ai_sorts

#initial sort is basicly the same as the random_draw_structure outcome

if __name__ == "__main__":
    team_schedules, possible_max_metric, league_teams, random_draw_structure, greedy_draw_structure, impoved_greedy_draw_structure = run_non_ai_sorts()
    #evolutionary(random_draw_structure, team_schedules, possible_max_metric, league_teams, plot=False)
    evolutionary(impoved_greedy_draw_structure, team_schedules, possible_max_metric, league_teams, plot=False)
