from datetime import date, datetime, timedelta
import os
from numpy import mat
from pandas.core.frame import DataFrame
import requests, json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

# Clear terminal output.
os.system('cls' if os.name == 'nt' else 'clear')

# Print iterations progress method.
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Create models for each player.
def predictive_player_model(player_id,model,independents):
    '''Creates predictive model for the player whose ID is passed.
        @params:
            player_id   - Required  : The player's unique ID (Int)
    '''

    history = df_playerHistory[df_playerHistory['element'] == player_id]
    
    # Gets next fixture's opponent.
    next_fixture = df_playerFixtures[df_playerFixtures['element'] == player_id].iloc[0]
    opponent_next_fixture = np.nan
    if next_fixture["is_home"]:
        opponent_next_fixture = df_teams[df_teams.id == next_fixture["team_a"]]
    else: 
        opponent_next_fixture = df_teams[df_teams.id == next_fixture["team_h"]]

    #if (player_id == 548) | (player_id == 484) | (player_id == 197) | (player_id == 308) | (player_id == 406) | (player_id == 235) | (player_id == 109):
        #print(history)

    predicted_score = 0
    if len(history) > 0:

        #for row in history.tail().iteritems():
            #print(model.params[0])
            #for coef in model.params:
                #print(str(name) + str(coef))
                #predicted_score += coef * row[name]

        df_next_fixture = pd.DataFrame(columns=df_playerHistory.columns)
        df_next_fixture = df_next_fixture.append({
            "strength_overall_home":opponent_next_fixture["strength_overall_home"],
            "strength_overall_away":opponent_next_fixture["strength_overall_away"],
            "strength_attack_home":opponent_next_fixture["strength_attack_home"],
            "strength_defence_home":opponent_next_fixture["strength_defence_home"],
            "strength_attack_away":opponent_next_fixture["strength_attack_away"],
            "strength_defence_away":opponent_next_fixture["strength_defence_away"],
            "ict_index":history["ict_index"].iloc[-1]       
        },ignore_index=True)

        df_next_fixture = df_next_fixture.fillna(0)
        # Gets total predicted score from model equation.
        index = 0
        #for coef in model.params:
            #predicted_score += coef * df_next_fixture[independent_vars[index]]
            #index += 1
        predicted_score = model.predict(history[independents])
        name = df_players[df_players['id_player'] == player_id]['web_name']
    return predicted_score


# base url for all FPL API endpoints
base_url = 'https://fantasy.premierleague.com/api/'

# get data from bootstrap-static endpoint
r = requests.get(base_url + 'bootstrap-static/').json()

# Get data from fixtures endpoint
r_fixtures = requests.get(base_url + 'fixtures/').json()

# Create events and fixtures dataframes.
df_fixtures = pd.json_normalize(r_fixtures)
df_events = pd.json_normalize(r['events'])

# Next gameweek.
next_gameweek = df_events.loc[df_events["is_next"] == True].id.iloc[0]

# Create team dataframe and reduces columns.
df_teams = pd.json_normalize(r['teams'])
df_teams = df_teams[['id', 'draw', 'form', 'loss', 'name', 'played', 'points',
       'position', 'strength', 'team_division', 'unavailable',
       'win', 'strength_overall_home', 'strength_overall_away',
       'strength_attack_home', 'strength_attack_away', 'strength_defence_home',
       'strength_defence_away'
    ]]

# Updates unfinished team data.
'''df_fixtures_finished = df_fixtures[df_fixtures.finished == True]
for team in df_teams:
    team["played"] = df_fixtures_finished.loc[(df_fixtures["team_h"] == team["id"] | df_fixtures["team_a"] == team["id"])].size
    for fixture in df_fixtures_finished:
        if team["id"] == fixture["team_h"]:
            if fixture["team_h_"]
        if team["id"] == fixture["team_a"]:'''

# Create a player dataframe and reduces columns.
df_players = pd.json_normalize(r['elements'])
df_players = df_players[['chance_of_playing_next_round', 'chance_of_playing_this_round',
       'element_type', 'ep_next', 'ep_this', 'event_points', 'first_name', 'form', 'id',
       'now_cost', 'photo', 'points_per_game', 'second_name', 'selected_by_percent',
       'status', 'team', 'total_points',
       'value_form', 'value_season', 'web_name', 'minutes', 'goals_scored',
       'assists', 'clean_sheets', 'goals_conceded', 'own_goals',
       'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards',
       'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat','ict_index'
    ]]
df_players['value'] = df_players.value_season.astype(float)
df_players['value_form'] = df_players.value_form.astype(float)

# Map position type info.
df_players['position'] = df_players.element_type.map(pd.json_normalize(r['element_types']).set_index('id').singular_name)

# Map team name.
df_players['team'] = df_players.team.map(df_teams.set_index('id').name)
df_fixtures['team_h'] = df_fixtures.team_h.map(df_teams.set_index('id').name)
df_fixtures['team_a'] = df_fixtures.team_a.map(df_teams.set_index('id').name)

# Drop unnecessary columns.
df_players = df_players.drop(axis='columns', labels={'value_season','element_type'})

# Only analyzes players that are available, have news, or are doubtful. Discards if a player is unavailable, injured, or suspended.
df_players = df_players[(df_players['status'] == 'a') | (df_players['status'] == 'n') | (df_players['status'] == 'd')]

# Removes players that have 0 points. 
df_players = df_players.loc[df_players.total_points > 0]

# Team value pivot table.
print(df_players.pivot_table(index='team',values='value',aggfunc=np.mean).reset_index().sort_values('value', ascending=False))

# Value by position.
print(df_players.pivot_table(index='position',values='value',aggfunc=np.mean).reset_index().sort_values('value', ascending=False))

# Function that gets all gameweek history for a given player Id.
def get_player_season_data(player_id):
    '''Get all gameweek info for a given player_id'''
    
    # send GET request to
    # https://fantasy.premierleague.com/api/element-summary/{PID}/
    r = requests.get(
            base_url + 'element-summary/' + str(player_id) + '/'
    ).json()
    
    return r

# Initializes variables for iteration.
i = 1
count = len(df_players.index)
playerHistory = []
playerFixtures = []
df_playerHistory = np.nan
df_playerFixtures = np.nan

useLiveData = False
if useLiveData:
    # Loops through player data frame and creates new dataframe purely with player gameweek history.
    for player_id in df_players['id']: 
        printProgressBar(i, count, "Getting player data ")
        playerData = get_player_season_data(player_id)

        if len(playerData['history']) > 0:
            for match in playerData['history']:
                playerHistory.append(match)
        if len(playerData['fixtures']) > 0:
            for match in playerData['fixtures']:
                match['element'] = player_id
                playerFixtures.append(match)
        i += 1

    # Creates dataframes for player history and player fixtures.
    df_playerHistory = pd.json_normalize(playerHistory)
    df_playerFixtures = pd.json_normalize(playerFixtures)

    # Add player position column.
    df_playerHistory = pd.merge(
        left=df_playerHistory,
        right=df_players[["id","position"]],
        left_on='element',
        right_on='id'
    )

    # Add fixture difficulty.
    df_playerHistory = pd.merge(
        left=df_playerHistory,
        right=df_fixtures[["id","team_h_difficulty","team_a_difficulty"]],
        left_on="fixture",
        right_on="id"      
    )

    # Converts 'Churn' with Yes and No values being 1 and 0, respectively.
    df_playerHistory = pd.get_dummies(df_playerHistory, drop_first=True, columns={"was_home"})
    df_playerHistory.insert(loc=df_playerHistory.columns.get_loc("was_home_True"), column="was_home", value=df_playerHistory.was_home_True)
    df_playerHistory.drop(axis="columns", labels="was_home_True")

    # Adds opponent_difficulty column based on whether the player was home or away.
    df_playerHistory["opponent_difficulty"] = df_playerHistory.apply(lambda row : row["team_h_difficulty"] if row["was_home"] == False else row["team_a_difficulty"], axis = 1)
    df_playerHistory = df_playerHistory.drop(axis='columns', labels={'team_h_difficulty','team_a_difficulty','id_x','id_y'})

    # Calculates the form the player was in at each gameweek.
    df_playerHistory["form"] = df_playerHistory.apply(lambda pl : 
        df_playerHistory[(df_playerHistory["element"] == pl["element"]) & (pd.to_datetime(df_playerHistory["kickoff_time"]) < pd.to_datetime(pl["kickoff_time"])) & (pd.to_datetime(df_playerHistory["kickoff_time"]) > (pd.to_datetime(pl["kickoff_time"]) - timedelta(days=30)))]["total_points"].mean(), axis = 1
    )

    # Fills null form with 0.
    df_playerHistory["form"] = df_playerHistory["form"].fillna(0)
    print(df_playerHistory.columns)
    # Only analyzes records where players actually played.
    df_playerHistory = df_playerHistory[df_playerHistory["minutes"] > 0]

    # Updates csv's.
    df_playerHistory.to_csv("history.csv")
    df_playerFixtures.to_csv("fixtures.csv")
else:
    df_playerHistory = pd.read_csv("history.csv")
    df_playerFixtures = pd.read_csv("fixtures.csv")

# Splits up dataframes into categories of goalkeeper, defender, midfielder, and forward.
df_playerHistory_goalkeeper = df_playerHistory[df_playerHistory['position'] == 'Goalkeeper']
df_playerHistory_defender = df_playerHistory[df_playerHistory['position'] == 'Defender']
df_playerHistory_midfielder = df_playerHistory[df_playerHistory['position'] == 'Midfielder']
df_playerHistory_forward = df_playerHistory[df_playerHistory['position'] == 'Forward']

# Creates OLS Multiple regression model.
model_goalkeeper = sm.OLS(df_playerHistory_goalkeeper["total_points"],df_playerHistory_goalkeeper[["opponent_difficulty","was_home","form"]]).fit()
model_defender = sm.OLS(df_playerHistory_defender["total_points"],df_playerHistory_defender[["opponent_difficulty","was_home","form"]]).fit()
model_midfielder = sm.OLS(df_playerHistory_midfielder["total_points"],df_playerHistory_midfielder[["opponent_difficulty","was_home","form"]]).fit()
model_forward = sm.OLS(df_playerHistory_forward["total_points"],df_playerHistory_forward[["opponent_difficulty","was_home","form"]]).fit()

print("\n\nGoalkeeper Model\n" + str(model_goalkeeper.summary()))
print("\n\nDefender Model\n" + str(model_defender.summary()))
print("\n\nMidfielder Model\n" + str(model_midfielder.summary()))
print("\n\nForward Model\n" + str(model_forward.summary()))

# Show message that player data updates are complete.
print("Finished getting player data.")

# Loops through player data frame and gets their predictive score for the next match.
df_players['next_week_predicted_score'] = np.nan

for index, row in df_players.iterrows():
    #printProgressBar(index, count, "Analyzing player data for next gameweek ")
    predicted = np.nan

    # Get next fixture(s) for player.
    next_fixture = df_fixtures.loc[(df_fixtures["event"] == next_gameweek) & ((df_fixtures["team_h"] == row["team"]) | (df_fixtures["team_a"] == row["team"]))]
    next_fixture = next_fixture.assign(form=float(row["form"]))
    next_fixture["opponent_difficulty"] = next_fixture.apply(lambda x : x["team_h_difficulty"] if row["team"] == x["team_a"] else x["team_a_difficulty"], axis=1)
    next_fixture["was_home"] = next_fixture.apply(lambda x : 1 if row["team"] == x["team_h"] else 0, axis=1)
    
    if len(next_fixture) == 1:
        if row["position"] == "Goalkeeper":
            predicted = model_goalkeeper.predict(next_fixture[["opponent_difficulty","was_home","form"]])
        if row["position"] == "Defender":
            predicted = model_defender.predict(next_fixture[["opponent_difficulty","was_home","form"]])
        if row["position"] == "Midfielder":
            predicted = model_midfielder.predict(next_fixture[["opponent_difficulty","was_home","form"]])
        if row["position"] == "Forward":
            predicted = model_forward.predict(next_fixture[["opponent_difficulty","was_home","form"]])
    if len(next_fixture) > 1:
        predicted = 0

    df_players.at[index, 'next_week_predicted_score'] = predicted

# Create squad based on highest total predicted points.
next_week_picks = []
players_sorted = df_players.sort_values(by=['next_week_predicted_score'], ascending=False)
for j, pl in players_sorted.iterrows():
    if (pl['position'] == 'Goalkeeper') & (sum(pk['position'] == 'Goalkeeper' for pk in next_week_picks) == 0):
        next_week_picks.append(pl)
    if (pl['position'] == 'Defender') & (sum(pk['position'] == 'Defender' for pk in next_week_picks) < 5):
        next_week_picks.append(pl)
    if (pl['position'] == 'Midfielder') & (sum(pk['position'] == 'Midfielder' for pk in next_week_picks) < 5):
        next_week_picks.append(pl)
    if (pl['position'] == 'Forward') & (sum(pk['position'] == 'Forward' for pk in next_week_picks) < 3):
        next_week_picks.append(pl)

    # Breaks loop if full squad is chosen.
    if len(next_week_picks) == 50:
        break

# Prints output.
for pick in next_week_picks:
    print(pick['first_name'] + " " + pick['second_name'] + ': ' + pick['team'] + '; Total: ' + str(round(pick['next_week_predicted_score'],2)) + '; id: ' + str(pick['id']))

