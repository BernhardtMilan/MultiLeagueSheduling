from evolutionary import evolutionary
from non_ai_sorts import run_non_ai_sorts
from benchmark_adaptive_tabu_search import ATS

team_schedules, possible_max_metric, league_teams, _, _, impoved_greedy_draw_data = run_non_ai_sorts()
_, ats_out_draw, _, _, _ = ATS(impoved_greedy_draw_data[0], team_schedules, league_teams, plot=True)
_, _, _, _, _ = evolutionary(ats_out_draw, team_schedules, possible_max_metric, league_teams, plot=True)