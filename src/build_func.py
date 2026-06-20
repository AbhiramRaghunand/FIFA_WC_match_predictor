import joblib
from collections import defaultdict
import pandas as pd

model_state=joblib.load(r'C:\ABHIRAM\projects\FifaWCPredictor\models\model_state.pkl')

for dictionary in model_state:
    if dictionary=='league_goals' or dictionary=='elo_ratings': continue
    model_state[dictionary]=defaultdict(list,model_state[dictionary])

def get_elo(team):
    return model_state['elo_ratings'].get(team,1500)

def get_recent_form(team,n=10):
    history=model_state['team_history'][team]
    if len(history)==0:
        return 0.5
    recent=history[-n:]
    return sum(recent)/len(recent)

def get_pair_key(team_a,team_b):
    return tuple(sorted([team_a,team_b]))

def get_h2h_form(team_home,team_away):
    key=get_pair_key(team_home,team_away)
    history=model_state['h2h_history'][key]
    if len(history)==0:
        return 0.5
    if team_home==key[0]:
        return sum(history)/len(history)
    else:
        return (len(history)-sum(history))/len(history)
    
def get_recent_goal_avg(team,history_dict,n=10):
    league_goals=model_state['league_goals']
    history=history_dict[team]
    if len(history) == 0:
        if len(league_goals) == 0:
            return 1.5
        return sum(league_goals) / len(league_goals)
    recent=history[-n:]
    return sum(recent)/len(recent)


def build_feature(team_home,team_away,neutral):
    elo_home=get_elo(team_home)
    elo_away=get_elo(team_away)

    form_home=get_recent_form(team_home)
    form_away=get_recent_form(team_away)

    h2h_home=get_h2h_form(team_home,team_away)

    goals_scored_home=get_recent_goal_avg(team_home,model_state['goal_scored_history'])
    goals_conceded_home=get_recent_goal_avg(team_home,model_state['goal_conceded_history'])

    goals_scored_away=get_recent_goal_avg(team_away,model_state['goal_scored_history'])
    goals_conceded_away=get_recent_goal_avg(team_away,model_state['goal_conceded_history'])


    features=[elo_home,elo_away, neutral,form_home,form_away, h2h_home,
              goals_scored_home,goals_conceded_home,goals_scored_away,goals_conceded_away]
    
    
    return pd.DataFrame([features],columns=['elo_home', 'elo_away', 'neutral',
                                            'form_home', 'form_away', 'h2h_home',
                                            'home_avg_goals', 'home_avg_conceded', 'away_avg_goals',
                                            'away_avg_conceded'])

# build_feature("Brazil","Argentina",neutral=True)
