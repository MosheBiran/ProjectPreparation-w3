import sqlite3
from aifc import Error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


scaler = MinMaxScaler()

path = "C:\\Users\\Daniel\\Downloads\\archive\\"



def init():
    """
    The Init And Building The Data From The Model Training And Testing
    :return: The Train Data , Test Data
    """

    """--------------------------------- SQL Connection And Queries ------------------------------------"""

    database = path + "database.sqlite"

    # create a database connection
    conn = create_connection(database)
    cursor = conn.cursor()

    # create DF
    match_Data_DF, team_Attr_Data_DF, teams_Data_DF, players_Data_Attr_DF, match_Players_Data_DF = sqlQuery(conn)


    # match_Data_DF = addLastMatchesGoals(match_Data_DF)


    """--------------------------------- Merging All The DataFrames Into One ------------------------------------"""

    Players_Attr_avg = dataframe_filter_players(match_Players_Data_DF, players_Data_Attr_DF)

    match_Data_DF_copy = match_Data_DF.copy()

    matchWithTeamAttributes_df = mergeMatchWithTeamAttribute(match_Data_DF, team_Attr_Data_DF)

    """--------------------------------- Adding New Features ------------------------------------"""

    # Label : Result Of The Match Per The Home Team
    matchWithTeamAttributes_df = addingResultFeature(matchWithTeamAttributes_df)

    # Feature : Winning Percentage Past Games
    matchWithTeamAttributes_df = addWinPresentInLastGames(matchWithTeamAttributes_df, match_Data_DF_copy)

    # Merging Data
    matchWithTeamAttributes_df = pd.merge(matchWithTeamAttributes_df, Players_Attr_avg, how='inner', left_on=['home_team_api_id', 'away_team_api_id', 'season', 'date'], right_on=['home_team_api_id', 'away_team_api_id', 'season', 'date'])

    # Feature : Mean Goals In The Last Season
    matchWithTeamAttributes_df = AddMeanGoalsLastSeason(matchWithTeamAttributes_df)

    # Feature : Team Attributes Ratio -> Home \ Away
    matchWithTeamAttributes_df = addTeamAttribute_Ratio(matchWithTeamAttributes_df)


    """--------------------------------- Shahar Function ------------------------------------"""

    # goals_df = matchWithTeamAttributes_df.apply(lambda x: get_last_matches(x, matchWithTeamAttributes_df), axis=1)
    # matchWithTeamAttributes_df = pd.merge(matchWithTeamAttributes_df, goals_df, how='left', left_on=['match_api_id'], right_on=['match_api_id'])

    """--------------------------------------------------------------------------------------"""

    """--------------------------------- Splitting The Data  ------------------------------------"""

    # Season For The Model Train
    trainData_before_Rename_Col = matchWithTeamAttributes_df.loc[(matchWithTeamAttributes_df['season'].isin(["2012/2013", "2013/2014", "2014/2015"]))]

    # Season 2015/2016 For Testing The Model
    testData_before_Rename_Col = matchWithTeamAttributes_df.loc[(matchWithTeamAttributes_df['season'].isin(["2015/2016"]))]

    """--------------------------------- Final Clearing And Organize Of The Data ------------------------------------"""

    # Rename The Features  - _x -> home | _y -> away
    trainData = remove_x_y(trainData_before_Rename_Col)
    testData = remove_x_y(testData_before_Rename_Col)

    # Deleting The UnNeeded Features
    trainData = clearUnusedFeatures(trainData)
    testData = clearUnusedFeatures(testData)

    # # Converting The Values Into Numbers
    # trainData = convertValuesStringToInt(trainData.copy())
    # testData = convertValuesStringToInt(testData.copy())

    # # Fit And Transform The Data
    # scaler.fit_transform(trainData.to_numpy())
    # scaler.fit_transform(testData.to_numpy())

    cursor.close()
    conn.close()

    return trainData, testData



def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn



def sqlQuery(conn):
    """
    The SQL Queries For Getting The Data From The SQL Database
    :param conn: The Connection
    :return: The SQL Data As DataFrames - Match_DF, TeamAttributes_DF, Teams_DF
    """
    data_matchDF = pd.read_sql_query(
        'SELECT match_api_id, home_team_api_id,away_team_api_id,season,date,home_team_goal,away_team_goal from Match', conn)

    data_Team_AttrDF = pd.read_sql_query(
        'SELECT team_api_id,date,buildUpPlaySpeed,buildUpPlayPassing,chanceCreationPassing, chanceCreationCrossing, '
        'chanceCreationShooting, defencePressure,defenceAggression, defenceTeamWidth from Team_Attributes',
        conn)

    data_Team = pd.read_sql_query('SELECT team_api_id, team_long_name from Team', conn)

    # data_Players_AttrDF = pd.read_sql_query(
    #     'SELECT player_api_id,date,overall_rating from Player_Attributes',
    #     conn)
    # data_Players_AttrDF = pd.read_sql_query(
    #     'SELECT player_api_id,date,overall_rating,potential,free_kick_accuracy,gk_kicking,shot_power from Player_Attributes',
    #     conn)
    data_Players_AttrDF = pd.read_sql_query(
        'SELECT player_api_id,date,overall_rating,potential,free_kick_accuracy,gk_kicking,shot_power,crossing,finishing,heading_accuracy,short_passing,volleys,dribbling,curve,long_passing,ball_control,acceleration,positioning,sprint_speed,aggression,long_shots from Player_Attributes',
        conn)

    data_matchDF_players = pd.read_sql_query(
        'SELECT home_team_api_id,away_team_api_id,season,date, home_player_1, home_player_2, home_player_3, '
        'home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, '
        'home_player_9, home_player_10, home_player_11 , away_player_1, away_player_2, away_player_3, away_player_4, '
        'away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11 '
        'from '
        'Match', conn)
    return data_matchDF, data_Team_AttrDF, data_Team, data_Players_AttrDF, data_matchDF_players



def dataframe_filter_players(data_match_players_df, player_attr_df):
    """
    :param data_match_players_df:Another frame containing player ID and that player attributes
    :param player_attr_df:data frame containing all  matches between 2 teams include  the season and date they play against each other
    :return:a new data frame that contains the team numbers along with the player average attributes in that season .
    """

    # Clearing the date from day and month
    data_match_players_df['date'] = data_match_players_df['date'].str.slice(stop=4)
    player_attr_df['date'] = player_attr_df['date'].str.slice(stop=4)

    # player_attr_df = player_attr_df.groupby(['player_api_id', 'date'], as_index=False)['overall_rating'].mean()
    player_attr_df = player_attr_df.groupby(['player_api_id', 'date'], as_index=False).mean(
        ['overall_rating', 'potential', 'free_kick_accuracy', 'gk_kicking', 'shot_power'])

    HomeAndAwayTeam_player_attr_mean_df = data_match_players_df[
        ['home_team_api_id', 'away_team_api_id', 'season', 'date']].copy()

    HomeAndAwayTeam_player_attr_mean_df = data_match_players_df[
        ['home_team_api_id', 'away_team_api_id', 'season', 'date']].copy()

    """--------------------------------- *** Moshe *** ------------------------------------"""


    # player_attr_df['potential_growth'] = player_attr_df.potential - player_attr_df.overall_rating
    # player_attr_df['overall_rating_bool'] = player_attr_df.apply(lambda x: 1 if x['overall_rating'] > 80 else 0, axis=1)
    # player_attr_df['potential_growth_bool'] = player_attr_df.apply(lambda x: 1 if x['potential_growth'] > 4 else 0,
    #                                                                axis=1)

    """---------------------------------***************------------------------------------"""


    for col in data_match_players_df.columns:
        if "_player_" in col:
            suffix = ("_home_", "_home_")
            if "away_" in col:
                suffix = ("_away_", "_away_")
            data_match_players_df = pd.merge(data_match_players_df, player_attr_df, how='left', left_on=['date', col],
                                             right_on=['date', 'player_api_id'], suffixes=suffix)
            del data_match_players_df[col]

    # removing all columns that are not relevant.
    data_match_players_df = data_match_players_df.drop(
        [col for col in data_match_players_df.columns if 'player_api_id' in col], axis=1)



    """--------------------------------- Overall Rating ------------------------------------"""


    # Creating a list of all columns that relevant to that specific team mean.
    home_col_mean_lst = [col for col in data_match_players_df.columns if 'overall_rating_home_' in col]
    away_col_mean_lst = [col for col in data_match_players_df.columns if 'overall_rating_away_' in col]
    HomeAndAwayTeam_player_attr_mean_df['home_player_overall_rating_mean'] = data_match_players_df[home_col_mean_lst].mean(
        1) / 100
    HomeAndAwayTeam_player_attr_mean_df['away_player_overall_rating_mean'] = data_match_players_df[away_col_mean_lst].mean(
        1) / 100

    # Ratio
    # HomeAndAwayTeam_player_attr_mean_df['players_rating'] = data_match_players_df[home_col_mean_lst].mean(1) / data_match_players_df[away_col_mean_lst].mean(1)



    """--------------------------------- *** Moshe *** ------------------------------------"""


    # """--------------------------------- potential_growth_bool ------------------------------------"""
    #
    # # Creating a list of all columns that relevant to that specific team mean.
    # home_col_mean_lst = [col for col in data_match_players_df.columns if 'potential_growth_bool_home_' in col]
    # away_col_mean_lst = [col for col in data_match_players_df.columns if 'potential_growth_bool_away_' in col]
    # HomeAndAwayTeam_player_attr_mean_df['home_potential_growth_bool'] = data_match_players_df[home_col_mean_lst].mean(
    #     1)
    #
    # """--------------------------------- overall_rating_bool ------------------------------------"""
    #
    # # Creating a list of all columns that relevant to that specific team mean.
    # home_col_mean_lst = [col for col in data_match_players_df.columns if 'overall_rating_bool_home_' in col]
    # away_col_mean_lst = [col for col in data_match_players_df.columns if 'overall_rating_bool_away_' in col]
    # HomeAndAwayTeam_player_attr_mean_df['home_overall_rating_bool'] = data_match_players_df[home_col_mean_lst].mean(
    #     1) * 10
    # HomeAndAwayTeam_player_attr_mean_df['away_overall_rating_bool'] = data_match_players_df[away_col_mean_lst].mean(
    #     1) * 10
    #
    """---------------------------------***************------------------------------------"""

    """--------------------------------- All Other Features ------------------------------------"""
    """-------------------------------- mixed ------------------------------------"""
    # home_col_mean_lst = [
    #                     col for col in data_match_players_df.columns
    #                      if 'potential_home_' in col or 'free_kick_accuracy_home_' in col or 'gk_kicking_home_' in col
    #                      or 'shot_power_home_' in col or 'crossing_home_' in col or 'finishing_home_' in col or 'heading_accuracy_home_' in col
    #                      or 'short_passing_home_' in col or 'volleys_home_' in col or 'dribbling_home_' in col or 'curve_home_' in col
    #                      or 'long_passing_home_' in col or 'ball_control_home_' in col or 'acceleration_home_' in col or
    #                      'positioning_home_' in col or 'sprint_speed_home_' in col or 'aggression_home_' in col or 'long_shots_home_' in col
    #                      ]
    # away_col_mean_lst = [
    #                     col for col in data_match_players_df.columns
    #                      if 'potential_away_' in col or 'free_kick_accuracy_away_' in col or 'gk_kicking_away_' in col
    #                      or 'shot_power_away_' in col or 'crossing_away_' in col or 'finishing_away_' in col or 'heading_accuracy_away_' in col
    #                      or 'short_passing_away_' in col or 'volleys_away_' in col or 'dribbling_away_' in col or 'curve_away_' in col
    #                      or 'long_passing_away_' in col or 'ball_control_away_' in col or 'acceleration_away_' in col or
    #                      'positioning_away_' in col or 'sprint_speed_away_' in col or 'aggression_away_' in col or 'long_shots_away_' in col
    #                      ]
    # HomeAndAwayTeam_player_attr_mean_df['home_against_away_player_attr_mixed_features_mean'] = (data_match_players_df[home_col_mean_lst] /
    #
    #                                                                          data_match_players_df[
    #                                                                               away_col_mean_lst]).mean(
    #     1) / 100
    # HomeAndAwayTeam_player_attr_mean_df['away_player_mixed_features_mean'] = (data_match_players_df[away_col_mean_lst]/data_match_players_df[home_col_mean_lst]).mean(
    #       1) / 100

    """--------------------------------- All Other Features ------------------------------------"""


    # """--------------------------------- All Other Features ------------------------------------"""
    # """------------------ Potential ------------------"""
    # if list_of_fields[0] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'potential_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'potential_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_pot_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_pot_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------free_kick_accuracy-----------"""
    # if list_of_fields[1] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'free_kick_accuracy_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'free_kick_accuracy_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_free_kick_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_free_kick_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------gk_kicking-------------------"""
    # if list_of_fields[2] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'gk_kicking_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'gk_kicking_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_gk_kicking_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_gk_kicking_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------shot_power-------------------"""
    # if list_of_fields[3] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'shot_power_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'shot_power_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_shot_power_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_shot_power_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------crossing-------------------"""
    # if list_of_fields[4] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'crossing_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'crossing_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_crossing_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_crossing_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------finishing-------------------"""
    # if list_of_fields[5] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'finishing_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'finishing_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_finishing_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_finishing_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------heading_accuracy-------------------"""
    # if list_of_fields[6] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'heading_accuracy_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'heading_accuracy_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_heading_accuracy_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_heading_accuracy_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------short_passing-------------------"""
    # if list_of_fields[7] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'short_passing_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'short_passing_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_short_passing_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_short_passing_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------volleys-------------------"""
    # if list_of_fields[8] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'volleys_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'volleys_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_volleys_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_volleys_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------dribbling-------------------"""
    # if list_of_fields[9] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'dribbling_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'dribbling_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_dribbling_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_dribbling_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------curve-------------------"""
    # if list_of_fields[10] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'curve_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'curve_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_curve_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_curve_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------long_passing-------------------"""
    # if list_of_fields[11] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'long_passing_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'long_passing_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_long_passing_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_long_passing_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------ball_control-------------------"""
    # if list_of_fields[13] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'ball_control_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'ball_control_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_ball_control_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_ball_control_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------acceleration-------------------"""
    # if list_of_fields[14] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'acceleration_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'acceleration_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_acceleration_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_acceleration_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------positioning-------------------"""
    # if list_of_fields[15] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'positioning_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'positioning_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_positioning_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_positioning_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------sprint_speed-------------------"""
    # if list_of_fields[16] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'sprint_speed_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'sprint_speed_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_sprint_speed_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_sprint_speed_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------aggression-------------------"""
    # if list_of_fields[18] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'aggression_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'aggression_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_aggression_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_aggression_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    # """------------------long_shots-------------------"""
    # if list_of_fields[19] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'long_shots_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'long_shots_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_long_shots_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_long_shots_mean'] = data_match_players_df[away_col_mean_lst].mean(
    #         1) / 100
    #
    #
    #
    # """--------------------------------- All Other Features ------------------------------------"""

    return HomeAndAwayTeam_player_attr_mean_df



def mergeMatchWithTeamAttribute(data_match_df, data_team_attr_df):
    """
    Merging The Data Of:
    -The Matches With
    -The Data Attributes Of Each Team Playing in Each Match
    ** Without Null **
    :param data_match_df: The Data about the matches
    :param data_team_attr_df: The Data about the team and there attributes
    :return: The Data merged with the matches and the attributes of the teams in the match - ** Without Null **
    """
    # Save the detailed date
    data_match_df['date_info'] = data_match_df['date']

    # Clearing the date from day and month
    data_match_df['date'] = data_match_df['date'].str.slice(stop=4)
    data_team_attr_df['date'] = data_team_attr_df['date'].str.slice(stop=4)

    # sorting by relevant col
    data_matchDF = data_match_df.sort_values(by=['home_team_api_id', 'away_team_api_id', 'date'])
    data_Team_AttrDF = data_team_attr_df.sort_values(by=['team_api_id', 'date'])

    # merging first by ['date', 'home_team_api_id'] and again by ['date', 'away_team_api_id']
    new_df_inner = pd.merge(data_matchDF, data_Team_AttrDF, how='inner', left_on=['date', 'home_team_api_id'],
                            right_on=['date', 'team_api_id'])
    new_df_inner = pd.merge(new_df_inner, data_Team_AttrDF, how='inner', left_on=['date', 'away_team_api_id'],
                            right_on=['date', 'team_api_id'])

    new_df_inner = new_df_inner.sort_values(by=['home_team_api_id', 'away_team_api_id', 'date'])

    return new_df_inner




def addingResultFeature(dataBeforeResult):
    """
    Add Result Label For Each Game - According To The Home Team
    "home_team_goals" - "away_team_goals" =
     2 = Win
     1 = Draw
     0 = Lose
    :param dataBeforeResult: The Dataframe With The Matches Information
    :return:The Data After Adding The Label "Result"
    """

    """---------------------------------* Win - Lose *------------------------------------"""


    # Adding a column of binary representation win loss and draw.
    conditions = [dataBeforeResult["home_team_goal"] > dataBeforeResult["away_team_goal"],
                  dataBeforeResult["home_team_goal"] <= dataBeforeResult["away_team_goal"]]

    choices = [2, 0]
    dataBeforeResult["result"] = np.select(conditions, choices, default=np.nan)
    dataBeforeResult["result"] = dataBeforeResult["result"].astype(int)


    """---------------------------------* Win - Draw - Lose *------------------------------------"""


    # Adding a column of binary representation win loss and draw.
    # conditions = [dataBeforeResult["home_team_goal"] > dataBeforeResult["away_team_goal"],
    #               dataBeforeResult["home_team_goal"] < dataBeforeResult["away_team_goal"],
    #               dataBeforeResult["home_team_goal"] == dataBeforeResult["away_team_goal"]]
    #
    # choices = [2, 0, 1]
    # dataBeforeResult["result"] = np.select(conditions, choices, default=np.nan)
    # dataBeforeResult["result"] = dataBeforeResult["result"].astype(int)

    return dataBeforeResult



def addWinPresentInLastGames(dataBeforeFeature, match_Data_DF):
    """
    Add New Feature - The Percentage Of Wins In The Past Games For Each Team In Match
    :param dataBeforeFeature: The DataFrame With All Matches Information But The Full Date
    :param match_Data_DF: The Original DataFrame Of The Matches - With The Full Date - For Merge
    :return: The DataFrame - Matches With The New Feature For Each Team
    """

    """--------------------------------- Preparing The Data Before Calculating ------------------------------------"""

    dataBeforeFeature = pd.merge(dataBeforeFeature, match_Data_DF, how='inner', left_on=['home_team_api_id', 'away_team_api_id', 'season', 'match_api_id'],
                                 right_on=['home_team_api_id', 'away_team_api_id', 'season', 'match_api_id'])


    match_sorted_by_home = dataBeforeFeature.sort_values(by=['home_team_api_id', 'date_y'])
    match_sorted_by_home = match_sorted_by_home[['home_team_api_id', 'date_y', 'result', 'match_api_id']]

    match_sorted_by_away = dataBeforeFeature.sort_values(by=['away_team_api_id', 'date_y'])
    match_sorted_by_away = match_sorted_by_away[['away_team_api_id', 'date_y', 'result', 'match_api_id']]


    """--------------------------------- Home Team - Win Home Present ------------------------------------"""


    match_sorted_by_home.insert(len(match_sorted_by_home.columns), 'percentWinHome', match_sorted_by_home['result'].astype(float))
    inxOfWins = len(match_sorted_by_home.columns) - 1

    dataWithHomePresent = calculateWinPresent(match_sorted_by_home, inxOfWins)


    """--------------------------------- Away Team - Win Away Present ------------------------------------"""


    match_sorted_by_away.insert(len(match_sorted_by_away.columns), 'percentWinAway', match_sorted_by_away['result'].astype(float))
    inxOfWins = len(match_sorted_by_away.columns) - 1

    dataWithAwayPresent = calculateWinPresent(match_sorted_by_away, inxOfWins)

    """--------------------------------- Merging The Data With The New Feature ------------------------------------"""

    dataBeforeFeature = pd.merge(dataBeforeFeature, dataWithHomePresent, how='inner', right_on=['home_team_api_id', 'match_api_id'], left_on=['home_team_api_id', 'match_api_id'])
    dataBeforeFeature = pd.merge(dataBeforeFeature, dataWithAwayPresent, how='inner', right_on=['away_team_api_id', 'match_api_id'], left_on=['away_team_api_id', 'match_api_id'])


    """--------------------------------- Delete And Rename Features Names ------------------------------------"""

    if 'date_y' in dataBeforeFeature:
        del dataBeforeFeature['date_y']
    if 'home_team_goal_y' in dataBeforeFeature:
        del dataBeforeFeature['home_team_goal_y']
    if 'away_team_goal_y' in dataBeforeFeature:
        del dataBeforeFeature['away_team_goal_y']

    dataBeforeFeature.rename(columns={'date_x': 'date'}, inplace=True)
    dataBeforeFeature.rename(columns={'home_team_goal_x': 'home_team_goal'}, inplace=True)
    dataBeforeFeature.rename(columns={'away_team_goal_x': 'away_team_goal'}, inplace=True)


    return dataBeforeFeature



"""-------------*** Help Function ***-------------"""


def calculateWinPresent(matches_sortedBy_HorA, inxOfWins):
    """
    Help Function - Calculating The Win Percentage For Each Team In The Last Matches
    :param matches_sortedBy_HorA: The DataFrame Of The Matches - Sorted By: HomeID / AwayID And Date
    :param inxOfWins: Index Of The New Feature
    :return: The DataFrame With The Past Games Win Percentage
    """
    home_away = ''

    if 'home_team_api_id' in matches_sortedBy_HorA:
        home_away = 'Home'
    elif 'away_team_api_id' in matches_sortedBy_HorA:
        home_away = 'Away'


    # matches_sortedBy_HorA.insert(len(matches_sortedBy_HorA.columns), 'percentDraws' + home_away, matches_sortedBy_HorA['result'].astype(float))
    # inxOfDraws = len(matches_sortedBy_HorA.columns) - 1
    # counterOfDraws = 0

    matches_sortedBy_HorA.insert(len(matches_sortedBy_HorA.columns), 'percentLoses' + home_away, matches_sortedBy_HorA['result'].astype(float))
    inxOfLoses = len(matches_sortedBy_HorA.columns) - 1
    counterOfLoses = 0


    """------------------------------------------------------"""
    i = 0
    startIndex = 0
    counterOfWins = 0
    teamID = 0
    while i < len(matches_sortedBy_HorA):
        # Moved To a New TeamID - Reboot All Variables
        if matches_sortedBy_HorA.iloc[i, 0] != teamID:
            teamID = matches_sortedBy_HorA.iloc[i, 0]
            startIndex = i
            counterOfWins = 0
            """-------** Shahar **-------"""
            # counterOfDraws = 0
            counterOfLoses = 0
            if matches_sortedBy_HorA.iloc[i, 2] == 2:
                counterOfWins += 1
            # elif matches_sortedBy_HorA.iloc[i, 2] == 1:
            #     counterOfDraws += 1
            elif matches_sortedBy_HorA.iloc[i, 2] == 0:
                counterOfLoses += 1
            matches_sortedBy_HorA.iat[i, inxOfWins] = 0.33
            # matches_sortedBy_HorA.iat[i, inxOfDraws] = 0.33
            matches_sortedBy_HorA.iat[i, inxOfLoses] = 0.33

        else:
            matches_sortedBy_HorA.iat[i, inxOfWins] = counterOfWins / (i - startIndex)
            """-------** Shahar **-------"""
            # matches_sortedBy_HorA.iat[i, inxOfDraws] = counterOfDraws / (i - startIndex)
            matches_sortedBy_HorA.iat[i, inxOfLoses] = counterOfLoses / (i - startIndex)
            if matches_sortedBy_HorA.iloc[i, 2] == 2:
                counterOfWins += 1
                """-------** Shahar **-------"""
            # elif matches_sortedBy_HorA.iloc[i, 2] == 1:
            #     counterOfDraws += 1
            elif matches_sortedBy_HorA.iloc[i, 2] == 0:
                counterOfLoses += 1
        i += 1

    # Delete No Needed Features
    if 'result' in matches_sortedBy_HorA:
        del matches_sortedBy_HorA['result']
    if 'date_y' in matches_sortedBy_HorA:
        del matches_sortedBy_HorA['date_y']

    return matches_sortedBy_HorA





def AddMeanGoalsLastSeason(data_df):

    """--------------------------------- New ------------------------------------"""


    data_home_df = data_df.groupby(['home_team_api_id', 'season'], as_index=False)['home_team_goal'].mean()
    data_away_df = data_df.groupby(['away_team_api_id', 'season'], as_index=False)['away_team_goal'].mean()

    """--------------------------------- Other ------------------------------------"""


    data_home_df_other = data_df.groupby(['home_team_api_id', 'season'], as_index=False)['away_team_goal'].mean()
    data_away_df_other = data_df.groupby(['away_team_api_id', 'season'], as_index=False)['home_team_goal'].mean()


    # data_home_df_other['away_team_goal'] = data_home_df_other['away_team_goal']*(-1)
    # data_away_df_other['home_team_goal'] = data_away_df_other['home_team_goal']*(-1)

    copy_data_home_df_other = data_home_df_other.copy()
    copy_data_away_df_other = data_away_df_other.copy()

    """--------------------------------------------------------------------------"""


    copy_data_home_df = data_home_df.copy()
    copy_data_away_df = data_away_df.copy()
    seasons = pd.DataFrame(data_home_df['season'])
    seasons = seasons.groupby(['season'])

    new_seasons = []
    old_seasons = []
    i = 0
    for x in seasons:
        if i == 0:
            i += 1
            old_seasons.append(x[0])
            continue
        new_seasons.append(x[0])
        old_seasons.append(x[0])
    old_seasons.pop()

    copy_data_home_df = copy_data_home_df[copy_data_home_df.season != '2015/2016']
    copy_data_away_df = copy_data_away_df[copy_data_away_df.season != '2015/2016']

    copy_data_home_df = copy_data_home_df.replace(old_seasons, new_seasons)
    copy_data_away_df = copy_data_away_df.replace(old_seasons, new_seasons)


    """--------------------------------- Other ------------------------------------"""

    copy_data_home_df_other = copy_data_home_df_other[copy_data_home_df_other.season != '2015/2016']
    copy_data_away_df_other = copy_data_away_df_other[copy_data_away_df_other.season != '2015/2016']

    copy_data_home_df_other = copy_data_home_df_other.replace(old_seasons, new_seasons)
    copy_data_away_df_other = copy_data_away_df_other.replace(old_seasons, new_seasons)


    """----------------------------------------------------------------------------"""

    """--------------------------------- Both Sides - Home-Home | Away-Away | Home-Away | Away-Home ------------------------------------"""



    # home_normalAndOther = pd.merge(copy_data_home_df, copy_data_home_df_other, how='inner', left_on=['home_team_api_id', 'season'], right_on=['home_team_api_id', 'season'])
    # away_normalAndOther = pd.merge(copy_data_away_df, copy_data_away_df_other, how='inner', left_on=['away_team_api_id', 'season'], right_on=['away_team_api_id', 'season'])
    #
    #
    # home_normalAndOther.rename(columns={'home_team_goal': 'home_season_team_goal'}, inplace=True)
    # home_normalAndOther.rename(columns={'away_team_goal': 'away_other_season_team_goal'}, inplace=True)
    #
    # away_normalAndOther.rename(columns={'away_team_goal': 'away_season_team_goal'}, inplace=True)
    # away_normalAndOther.rename(columns={'home_team_goal': 'home_other_season_team_goal'}, inplace=True)
    #
    #
    # data_df = pd.merge(data_df, home_normalAndOther, how='left', left_on=['home_team_api_id', 'season'], right_on=['home_team_api_id', 'season'])
    # data_df = pd.merge(data_df, away_normalAndOther, how='left', left_on=['away_team_api_id', 'season'], right_on=['away_team_api_id', 'season'])


    """--------------------------------- Home-Home | Away-Away ------------------------------------"""


    data_df = pd.merge(data_df, copy_data_home_df, how='left', left_on=['home_team_api_id', 'season'], right_on=['home_team_api_id', 'season'])
    data_df = pd.merge(data_df, copy_data_away_df, how='left', left_on=['away_team_api_id', 'season'], right_on=['away_team_api_id', 'season'])


    data_df.rename(columns={'home_team_goal_x': 'home_team_goal'}, inplace=True)
    data_df.rename(columns={'away_team_goal_x': 'away_team_goal'}, inplace=True)

    data_df.rename(columns={'home_team_goal_y': 'home_season_team_goal'}, inplace=True)
    data_df.rename(columns={'away_team_goal_y': 'away_season_team_goal'}, inplace=True)


    """--------------------------------- Add Season Goals Class ------------------------------------"""


    conditions_percent = [data_df["home_season_team_goal"] >= data_df["away_season_team_goal"],
                          data_df["home_season_team_goal"] < data_df["away_season_team_goal"]]

    choices = [1, -1]
    data_df["seasonGoals_Class"] = np.select(conditions_percent, choices, default=np.nan)


    # print(data_df["seasonGoals_Class"].value_counts())


    # data_df['away_season_team_goal'] = data_df['away_season_team_goal']*(-1)



    """------------------------------------------------------------------------------"""


    # print(data_df.apply(lambda x: sum(x.isnull()), axis=0))
    data_df = data_df.dropna()
    # print(data_df.apply(lambda x: sum(x.isnull()), axis=0))


    """--------------------------------- Origin ------------------------------------"""
    #
    # data_home_df = data_df.groupby(['home_team_api_id', 'season'], as_index=False)['home_team_goal'].mean()
    # data_away_df = data_df.groupby(['away_team_api_id', 'season'], as_index=False)['away_team_goal'].mean()
    #
    #
    # data_df = pd.merge(data_df, data_home_df, how='left', left_on=['home_team_api_id', 'season'],
    #                    right_on=['home_team_api_id', 'season'])
    # data_df = pd.merge(data_df, data_away_df, how='left', left_on=['away_team_api_id', 'season'],
    #                    right_on=['away_team_api_id', 'season'])
    #
    # # data_df['goals_mean'] = np.floor(data_df['home_team_goal_y'] / data_df['away_team_goal_y'])
    # # del data_df['home_team_goal_y']
    # # del data_df['away_team_goal_y']
    #
    # data_df.rename(columns={'home_team_goal_x': 'home_team_goal'}, inplace=True)
    # data_df.rename(columns={'away_team_goal_x': 'away_team_goal'}, inplace=True)
    #
    # data_df.rename(columns={'home_team_goal_y': 'home_season_team_goal'}, inplace=True)
    # data_df.rename(columns={'away_team_goal_y': 'away_season_team_goal'}, inplace=True)
    #
    # # home_away_goals = pd.merge(data_home_df, data_away_df, how='outer', left_on=['home_team_api_id'], right_on=['away_team_api_id'])
    # # home_away_goals['goal'] = home_away_goals[['home_team_goal','away_team_goal']].mean(1)
    # # del home_away_goals['away_team_api_id']
    # # del home_away_goals['home_team_goal']
    # # del home_away_goals['away_team_goal']
    # #
    # # data_df = pd.merge(data_df, home_away_goals, how='left', left_on=['home_team_api_id'], right_on=['home_team_api_id'])
    # # data_df = pd.merge(data_df, home_away_goals, how='left', left_on=['away_team_api_id'], right_on=['home_team_api_id'])
    # # data_df['goals_mean'] = data_df['goal_x']/data_df['goal_y']
    # #
    # # data_df.rename(columns={'home_team_api_id_x': 'home_team_api_id'}, inplace=True)
    # #
    # # del data_df['goal_x']
    # # del data_df['goal_y']
    # # del data_df['home_team_api_id_y']

    """------------------------------------------------------------------------------"""

    return data_df




def addTeamAttribute_Ratio(data_df):

    """--------------------------------- New ------------------------------------"""

    data_df['overall_home_team_rating'] = data_df.loc[:, 'buildUpPlaySpeed_x':'defenceTeamWidth_x'].mean(axis=1) / data_df.loc[:, 'buildUpPlaySpeed_y':'defenceTeamWidth_y'].mean(axis=1)


    # data_df['overall_away_team_rating'] = data_df.loc[:, 'buildUpPlaySpeed_y':'defenceTeamWidth_y'].mean(axis=1)
    # data_df['overall_away_team_rating'] = data_df['overall_away_team_rating']*(-1)

    if 'buildUpPlaySpeed_x' in data_df:
        del data_df['buildUpPlaySpeed_x']
    if 'buildUpPlaySpeed_y' in data_df:
        del data_df['buildUpPlaySpeed_y']
    if 'buildUpPlayPassing_x' in data_df:
        del data_df['buildUpPlayPassing_x']
    if 'buildUpPlayPassing_y' in data_df:
        del data_df['buildUpPlayPassing_y']
    if 'chanceCreationPassing_x' in data_df:
        del data_df['chanceCreationPassing_x']
    if 'chanceCreationPassing_y' in data_df:
        del data_df['chanceCreationPassing_y']
    if 'chanceCreationCrossing_x' in data_df:
        del data_df['chanceCreationCrossing_x']
    if 'chanceCreationCrossing_y' in data_df:
        del data_df['chanceCreationCrossing_y']
    if 'chanceCreationShooting_x' in data_df:
        del data_df['chanceCreationShooting_x']
    if 'chanceCreationShooting_y' in data_df:
        del data_df['chanceCreationShooting_y']
    if 'defencePressure_x' in data_df:
        del data_df['defencePressure_x']
    if 'defencePressure_y' in data_df:
        del data_df['defencePressure_y']
    if 'defenceAggression_x' in data_df:
        del data_df['defenceAggression_x']
    if 'defenceAggression_y' in data_df:
        del data_df['defenceAggression_y']
    if 'defenceTeamWidth_x' in data_df:
        del data_df['defenceTeamWidth_x']
    if 'defenceTeamWidth_y' in data_df:
        del data_df['defenceTeamWidth_y']

    """--------------------------------- Origin ------------------------------------"""

    # data_df['buildUpPlaySpeed'] = data_df['buildUpPlaySpeed_x']/data_df['buildUpPlaySpeed_y']
    # del data_df['buildUpPlaySpeed_x']
    # del data_df['buildUpPlaySpeed_y']
    #
    # data_df['buildUpPlayPassing'] = data_df['buildUpPlayPassing_x']/data_df['buildUpPlayPassing_y']
    # del data_df['buildUpPlayPassing_x']
    # del data_df['buildUpPlayPassing_y']
    #
    # data_df['chanceCreationPassing'] = data_df['chanceCreationPassing_x']/data_df['chanceCreationPassing_y']
    # del data_df['chanceCreationPassing_x']
    # del data_df['chanceCreationPassing_y']
    #
    # data_df['chanceCreationCrossing'] = data_df['chanceCreationCrossing_x']/data_df['chanceCreationCrossing_y']
    # del data_df['chanceCreationCrossing_x']
    # del data_df['chanceCreationCrossing_y']
    #
    # data_df['chanceCreationShooting'] = data_df['chanceCreationShooting_x']/data_df['chanceCreationShooting_y']
    # del data_df['chanceCreationShooting_x']
    # del data_df['chanceCreationShooting_y']
    #
    # data_df['defencePressure'] = data_df['defencePressure_x']/data_df['defencePressure_y']
    # del data_df['defencePressure_x']
    # del data_df['defencePressure_y']
    #
    # data_df['defenceAggression'] = data_df['defenceAggression_x']/data_df['defenceAggression_y']
    # del data_df['defenceAggression_x']
    # del data_df['defenceAggression_y']
    #
    # data_df['defenceTeamWidth'] = data_df['defenceTeamWidth_x']/data_df['defenceTeamWidth_y']
    # del data_df['defenceTeamWidth_x']
    # del data_df['defenceTeamWidth_y']

    """------------------------------------------------------------------------------"""

    return data_df




def remove_x_y(dataBeforeRename):
    """
    Removing From The Features Names the "_x" And "_y" -> "home_" And "away_"
    :param dataBeforeRename: The Data that we want to convert
    :return: The Data After The Convert
    """
    for col in dataBeforeRename.columns:
        if '_x' == col[len(col) - 2:len(col)]:
            dataBeforeRename = dataBeforeRename.rename(
                columns={col: 'home_' + col[:len(col) - 2]}
                , inplace=False)
        if '_y' == col[len(col) - 2:len(col)]:
            dataBeforeRename = dataBeforeRename.rename(
                columns={col: 'away_' + col[:len(col) - 2]}
                , inplace=False)

    return dataBeforeRename




def clearUnusedFeatures(dataToClean):
    """
    Cleaning The Features That We No Longer Needed
    - After The Init And
    - Before The Train
    :param dataToClean: The Dataframe that we need to clear
    :return: The Dataframe after clean
    """
    if "home_team_goal" in dataToClean:
        del dataToClean["home_team_goal"]

    if "away_team_goal" in dataToClean:
        del dataToClean["away_team_goal"]

    if "season" in dataToClean:
        del dataToClean["season"]

    if "date" in dataToClean:
        del dataToClean["date"]

    if "home_team_api_id" in dataToClean:
        del dataToClean["home_team_api_id"]
    if "away_team_api_id" in dataToClean:
        del dataToClean["away_team_api_id"]

    if "match_api_id" in dataToClean:
        del dataToClean['match_api_id']

    if "date_info" in dataToClean:
        del dataToClean["date_info"]

    if "home_Away_LastMatchesGoals" in dataToClean:
        del dataToClean["home_Away_LastMatchesGoals"]

    if "away_Home_LastMatchesGoals" in dataToClean:
        del dataToClean["away_Home_LastMatchesGoals"]

    if "result" in dataToClean:
        dataToClean["Result"] = dataToClean["result"]
        del dataToClean["result"]


    # del new_df["home_percentHome"]
    # del new_df["home_percentAway"]
    # del new_df["away_percentHome"]
    # del new_df["away_percentAway"]

    # del new_df["away_whereBetter"]
    # del new_df["home_whereBetter"]

    # del new_df["home_season_team_goal"]
    # del new_df["away_season_team_goal"]
    # del new_df["home_other_season_team_goal"]
    # del new_df["away_other_season_team_goal"]


    return dataToClean




"""--------------------------------- Shahar Function ------------------------------------"""



def get_last_matches(match, data_df):
    x = 15
    y = 10
    # Define variables
    date = match.date_info
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id

    # Get last x matches of home and away team

    # Filter team matches from matches
    home_team_matches = data_df[(data_df['home_team_api_id'] == home_team) | (data_df['away_team_api_id'] == home_team)]
    home_last_matches = home_team_matches[home_team_matches.date_info < date].sort_values(by='date_info', ascending=False).iloc[0:x, :]

    away_team_matches = data_df[(data_df['away_team_api_id'] == away_team) | (data_df['home_team_api_id'] == away_team)]
    away_last_matches = away_team_matches[away_team_matches.date_info < date].sort_values(by='date_info', ascending=False).iloc[0:x, :]

    # Create goal variables
    home_goals = (int(home_last_matches.home_team_goal[home_last_matches.home_team_api_id == home_team].sum()) + int(home_last_matches.away_team_goal[home_last_matches.away_team_api_id == home_team].sum()))
    away_goals = (int(away_last_matches.home_team_goal[away_last_matches.home_team_api_id == away_team].sum()) + int(away_last_matches.away_team_goal[away_last_matches.away_team_api_id == away_team].sum()))
    home_goals_conceided = (int(home_last_matches.home_team_goal[home_last_matches.away_team_api_id == home_team].sum()) + int(home_last_matches.away_team_goal[home_last_matches.home_team_api_id == home_team].sum()))
    away_goals_conceided = (int(away_last_matches.home_team_goal[away_last_matches.away_team_api_id == away_team].sum()) + int(away_last_matches.away_team_goal[away_last_matches.home_team_api_id == away_team].sum()))



    # Create win variables
    total_wins_home = int(home_last_matches.home_team_goal[(home_last_matches.home_team_api_id == home_team) & (home_last_matches.home_team_goal > home_last_matches.away_team_goal)].count()) + int(home_last_matches.away_team_goal[(home_last_matches.away_team_api_id == home_team) & (home_last_matches.away_team_goal > home_last_matches.home_team_goal)].count())
    total_wins_away = int(away_last_matches.home_team_goal[(away_last_matches.home_team_api_id == away_team) & (away_last_matches.home_team_goal > away_last_matches.away_team_goal)].count()) + int(away_last_matches.away_team_goal[(away_last_matches.away_team_api_id == away_team) & (away_last_matches.away_team_goal > away_last_matches.home_team_goal)].count())
    total_lose_home = int(home_last_matches.home_team_goal[(home_last_matches.home_team_api_id == home_team) & (home_last_matches.home_team_goal < home_last_matches.away_team_goal)].count()) + int(home_last_matches.away_team_goal[(home_last_matches.away_team_api_id == home_team) & (home_last_matches.away_team_goal < home_last_matches.home_team_goal)].count())
    total_lose_away = int(away_last_matches.home_team_goal[(away_last_matches.home_team_api_id == away_team) & (away_last_matches.home_team_goal < away_last_matches.away_team_goal)].count()) + int(away_last_matches.away_team_goal[(away_last_matches.away_team_api_id == away_team) & (away_last_matches.away_team_goal < away_last_matches.home_team_goal)].count())
    total_draw_home = int(home_last_matches.home_team_goal[(home_last_matches.home_team_api_id == home_team) & (home_last_matches.home_team_goal == home_last_matches.away_team_goal)].count()) + int(home_last_matches.away_team_goal[(home_last_matches.away_team_api_id == home_team) & (home_last_matches.away_team_goal == home_last_matches.home_team_goal)].count())
    total_draw_away = int(away_last_matches.home_team_goal[(away_last_matches.home_team_api_id == away_team) & (away_last_matches.home_team_goal == away_last_matches.away_team_goal)].count()) + int(away_last_matches.away_team_goal[(away_last_matches.away_team_api_id == away_team) & (away_last_matches.away_team_goal == away_last_matches.home_team_goal)].count())


    # Create variables of 2 given teams
    home_and_away_matches = data_df[(data_df['home_team_api_id'] == home_team) & (data_df['away_team_api_id'] == away_team)]
    away_and_home_matches = data_df[(data_df['home_team_api_id'] == away_team) & (data_df['away_team_api_id'] == home_team)]
    total_home_and_away_matches = pd.concat([home_and_away_matches, away_and_home_matches])

    try:
        last_home_and_away_matches = total_home_and_away_matches[total_home_and_away_matches.date_info < date].sort_values(by='date_info', ascending=False).iloc[0:y, :]
    except:
        last_home_and_away_matches = total_home_and_away_matches[total_home_and_away_matches.date < date].sort_values(by='date_info', ascending=False).iloc[0:total_home_and_away_matches.shape[0], :]

        # Check for error in data
        if last_home_and_away_matches.shape[0] > y:
            print("Error in obtaining matches")


    total_wins_home_and_away = int(last_home_and_away_matches.home_team_goal[(last_home_and_away_matches.home_team_api_id == home_team) & (last_home_and_away_matches.home_team_goal > last_home_and_away_matches.away_team_goal)].count()) + int(last_home_and_away_matches.away_team_goal[(last_home_and_away_matches.away_team_api_id == home_team) & (last_home_and_away_matches.away_team_goal > last_home_and_away_matches.home_team_goal)].count())
    total_wins_away_and_home = int(last_home_and_away_matches.home_team_goal[(last_home_and_away_matches.home_team_api_id == away_team) & (last_home_and_away_matches.home_team_goal > last_home_and_away_matches.away_team_goal)].count()) + int(last_home_and_away_matches.away_team_goal[(last_home_and_away_matches.away_team_api_id == away_team) & (last_home_and_away_matches.away_team_goal > last_home_and_away_matches.home_team_goal)].count())



    # Define result data frame
    result = pd.DataFrame()

    # Define ID features
    result.loc[0, 'match_api_id'] = match.match_api_id

    # Create match features
    result.loc[0, 'home_team_goals_difference'] = home_goals
    result.loc[0, 'away_team_goals_difference'] = away_goals
    result.loc[0, 'home_team_goals_conceided'] = home_goals_conceided
    result.loc[0, 'games_won_home_team'] = total_wins_home
    result.loc[0, 'games_won_away_team'] = total_wins_away
    result.loc[0, 'games_lose_home_team'] = total_lose_home
    result.loc[0, 'games_draw_home_team'] = total_draw_home
    result.loc[0, 'games_draw_away_team'] = total_draw_away

    result.loc[0, 'games_against_won'] = total_wins_home_and_away
    result.loc[0, 'games_against_lost'] = total_wins_away_and_home

    # Return match features
    return result.loc[0]


"""---------------------------------------------------------------------"""



"""--------------------------------- Daniel Function ------------------------------------"""



def addLastMatchesGoals(match_Data_DF):
    """
    Add New Feature - Sum Of The Last 'gamesBack' Of Each Team In Match
    :param match_Data_DF: The DataFrame Of Matches
    :return: The DataFrame With The New Feature - LastMatchesGoals
    """

    """--------------------------------- Splitting To Home And Away------------------------------------"""


    match_sorted_by_home = match_Data_DF.sort_values(by=['home_team_api_id', 'date'])
    match_sorted_by_home = match_sorted_by_home[['home_team_api_id', 'date', 'home_team_goal']]

    match_sorted_by_away = match_Data_DF.sort_values(by=['away_team_api_id', 'date'])
    match_sorted_by_away = match_sorted_by_away[['away_team_api_id', 'date', 'away_team_goal']]


    """--------------------------------- Home Team Goals Calculate ------------------------------------"""


    match_sorted_by_home.insert(len(match_sorted_by_home.columns), 'Home_LastMatchesGoals', match_sorted_by_home['home_team_goal'])
    inxOfCol = len(match_sorted_by_home.columns) - 1

    homeAfterAdd = calculateHomeOrAwayGoals(match_sorted_by_home, inxOfCol, 'home_')


    """--------------------------------- Away Team Goals Calculate ------------------------------------"""


    match_sorted_by_away.insert(len(match_sorted_by_away.columns), 'Away_LastMatchesGoals', match_sorted_by_away['away_team_goal'])
    inxOfCol = len(match_sorted_by_away.columns) - 1

    awayAfterAdd = calculateHomeOrAwayGoals(match_sorted_by_away, inxOfCol, 'away_')



    """--------------------------------- Merge To Main Data ------------------------------------"""


    match_Data_DF = pd.merge(match_Data_DF, homeAfterAdd, how='inner',
                             left_on=['home_team_api_id', 'date', 'home_team_goal'],
                             right_on=['home_team_api_id', 'date', 'home_team_goal'])

    match_Data_DF = pd.merge(match_Data_DF, awayAfterAdd, how='inner',
                             left_on=['away_team_api_id', 'date', 'away_team_goal'],
                             right_on=['away_team_api_id', 'date', 'away_team_goal'])

    return match_Data_DF





def calculateHomeOrAwayGoals(sortedHomeOrAway, inxOfCol, home_away, gamesBack=3):
    """
    Help Function - Calculating the Number Of Goals In The Last 'gamesBack' Games Per Team
    :param sortedHomeOrAway: The DataFrame With The Team Games - Sorted By TeamID And Date
    :param inxOfCol: The Index Of The New Feature
    :param home_away: Which Team We Now - Home Or Away
    :param gamesBack: How Many Games Back Need To Calculate
    :return: The DataFrame With The New Feature - The Team With The Sum Of Last Games
    """
    i = 0
    indexesOfStarts = []
    teamID = 0
    while i < len(sortedHomeOrAway):
        if sortedHomeOrAway.iloc[i, 0] != teamID:
            indexesOfStarts.append(i)
            i += gamesBack
            teamID = sortedHomeOrAway.iloc[i, 0]

        sortedHomeOrAway.iat[i, inxOfCol] = sortedHomeOrAway[home_away+'team_goal'][i - gamesBack: i].sum()
        i += 1

    for i, indexNow in enumerate(indexesOfStarts):
        if i + 1 != len(indexesOfStarts):
            fillMean = np.floor(sortedHomeOrAway.iloc[indexNow + gamesBack:indexesOfStarts[i + 1], inxOfCol].mean())
        else:
            fillMean = np.floor(sortedHomeOrAway.iloc[indexNow + gamesBack:, inxOfCol].mean())
        for j in range(indexNow, indexNow + gamesBack):
            sortedHomeOrAway.iat[j, inxOfCol] = fillMean

    return sortedHomeOrAway







"""---------------------------------------- Functions With No Longer Use -------------------------------------------"""




def save2CSV(database_after_clean, file_path):
    """
    Save The Database As CSV File
    :param database_after_clean: The data that we want to save in CSV
    :param file_path: The Path where we want to save the database as CSV
    """
    database_after_clean.to_csv(file_path + "database_after_clean.csv")




def convertValuesStringToInt(data):
    """
    Convert The Values Of The Features - From String To Numeric
    :param data: The DataFrames Needed To Convert
    :return: The DataFrame With Numeric Values
    """
    le = preprocessing.LabelEncoder()
    for col in data.columns:
        if isinstance(data[col][0], str) and "name" not in col:
            # turn a string label into a number
            data[col] = le.fit_transform(data[col])

    return data



def margeWhereBetterWithMainData(df_2012_2013_2014_before, df_15_16_before):
    """
    Merging The Dataframe Of Where Better With The Main Dataframe
    :param df_2012_2013_2014_before: The Main Dataframe without "whereBetter"
    :param df_15_16_before: The Dataframe Of "whereBetter"
    :return: The Main Dataframe With The Feature "whereBetter"  - Train, Test
    """
    df_percent_win_12_13_14 = getWhereBetterHomeOrAway(df_2012_2013_2014_before)
    df_percent_win_15_16 = getWhereBetterHomeOrAway(df_15_16_before)

    df_2012_2013_2014 = pd.merge(df_2012_2013_2014_before, df_percent_win_12_13_14, how='inner',
                                 left_on=['home_team_api_id'],
                                 right_on=['team_api_id'])

    df_2012_2013_2014 = pd.merge(df_2012_2013_2014, df_percent_win_12_13_14, how='inner',
                                 left_on=['away_team_api_id'],
                                 right_on=['team_api_id'])

    df_15_16 = pd.merge(df_15_16_before, df_percent_win_15_16, how='inner',
                        left_on=['home_team_api_id'],
                        right_on=['team_api_id'])

    df_15_16 = pd.merge(df_15_16, df_percent_win_15_16, how='inner',
                        left_on=['away_team_api_id'],
                        right_on=['team_api_id'])

    df_2012_2013_2014 = remove_x_y(df_2012_2013_2014)
    df_15_16 = remove_x_y(df_15_16)

    return df_2012_2013_2014, df_15_16



def getWhereBetterHomeOrAway(new_df_with_name):
    """
    Calculate Where The Team Better Playing
    -At Home
    -Away
    -NeverMind
    :param new_df_with_name: The Data with all features we need to calculate
    :return: Dataframe With Team ID And The Result Where Better
    """
    df_home_team_win_sum = new_df_with_name.groupby(["home_team_api_id", "result"]).size().reset_index(
        name="wins_home_sum")
    df_home_team_win_sum = df_home_team_win_sum.loc[(df_home_team_win_sum['result'] == 2)]

    df_away_team_win_sum = new_df_with_name.groupby(["away_team_api_id", "result"]).size().reset_index(
        name="wins_away_sum")
    df_away_team_win_sum = df_away_team_win_sum.loc[(df_away_team_win_sum['result'] == 0)]

    df_home_team_total_count = new_df_with_name.groupby(["home_team_api_id"]).result.count().reset_index(
        name="home_count")

    df_away_team_total_count = new_df_with_name.groupby(["away_team_api_id"]).result.count().reset_index(
        name="away_count")

    del df_home_team_win_sum['result']
    del df_away_team_win_sum['result']

    df_home_team_win_sum = df_home_team_win_sum.merge(df_home_team_total_count, on=['home_team_api_id'], how='left')
    df_away_team_win_sum = df_away_team_win_sum.merge(df_away_team_total_count, on=['away_team_api_id'], how='left')

    df_away_team_win_sum['percentAway'] = df_away_team_win_sum[['wins_away_sum']].div(
        df_away_team_win_sum['away_count'], axis=0)
    df_home_team_win_sum['percentHome'] = df_home_team_win_sum[['wins_home_sum']].div(
        df_home_team_win_sum['home_count'], axis=0)

    df_home_team_win_sum = df_home_team_win_sum.rename(columns={'home_team_api_id': 'team_api_id'}, inplace=False)
    df_away_team_win_sum = df_away_team_win_sum.rename(columns={'away_team_api_id': 'team_api_id'}, inplace=False)

    df_percent_wim = pd.merge(df_home_team_win_sum, df_away_team_win_sum, how='outer', left_on=['team_api_id'],
                              right_on=['team_api_id'])

    df_percent_wim = df_percent_wim[['team_api_id', 'percentHome', 'percentAway']].copy()
    df_percent_wim['percentHome'].fillna(0, inplace=True)
    df_percent_wim['percentAway'].fillna(0, inplace=True)

    conditions_percent = [df_percent_wim["percentHome"] > df_percent_wim["percentAway"],
                          df_percent_wim["percentHome"] < df_percent_wim["percentAway"],
                          df_percent_wim["percentHome"] == df_percent_wim["percentAway"]]

    choices = ["Home", "Away", "NeverMind"]
    df_percent_wim["whereBetter"] = np.select(conditions_percent, choices, default=np.nan)
    return df_percent_wim




def resultToCategorical(dataToConvert):
    """
    Converting the result feature from int to categorical [2, 1, 0] -> ["Win", "Draw", "Lose"]
    :param dataToConvert: The Dataframe that needed to convert
    :return: The Dataframe after the convert
    """
    # Adding a column of binary representation win loss and draw.
    conditions = [dataToConvert["result"] == 2,
                  dataToConvert["result"] == 1,
                  dataToConvert["result"] == 0]

    choices = ["Win", "Draw", "Lose"]
    dataToConvert["result"] = np.select(conditions, choices, default=np.nan)
    dataToConvert["HomeTeamResult"] = dataToConvert["result"]
    del dataToConvert["result"]

    return dataToConvert



def addTeamNames(dataBeforeTeamNames, teams_Data_DF):
    """
    Add Into The Data The Teams Unique Names According The Team Api ID
    :param dataBeforeTeamNames: The Data that we want to add the teams names
    :param teams_Data_DF: The Data with the teams names
    :return: The Final Data With The Teams Names Inside
    """
    new_df_with_name = pd.merge(teams_Data_DF, dataBeforeTeamNames, how='inner', left_on=['team_api_id'],
                                right_on=['away_team_api_id'])

    new_df_with_name = pd.merge(teams_Data_DF, new_df_with_name, how='inner', left_on=['team_api_id'],
                                right_on=['home_team_api_id'])


    if 'team_api_id_x' in new_df_with_name:
        del new_df_with_name['team_api_id_x']
    if 'team_api_id_y' in new_df_with_name:
        del new_df_with_name['team_api_id_y']
    new_df_with_name = remove_x_y(new_df_with_name)

    return new_df_with_name




def mergeMatchWithTeamAttribute_WithNull(data_match_df, data_team_attr_df):
    """
    Merging The Data Of:
    -The Matches With
    -The Data Attributes Of Each Team Playing in Each Match
    ** With Null **
    :param data_match_df: The Data about the matches
    :param data_team_attr_df: The Data about the team and there attributes
    :return: The Data merged with the matches and the attributes of the teams in the match - With Null
    """  # Clearing the date from day and month
    data_match_df['date'] = data_match_df['date'].str.slice(stop=4)
    data_team_attr_df['date'] = data_team_attr_df['date'].str.slice(stop=4)

    # sorting by relevant col
    data_matchDF = data_match_df.sort_values(by=['home_team_api_id', 'away_team_api_id', 'date'])
    data_Team_AttrDF = data_team_attr_df.sort_values(by=['team_api_id', 'date'])

    # merging first by ['date', 'home_team_api_id'] and again by ['date', 'away_team_api_id']
    new_df_outer = pd.merge(data_matchDF, data_Team_AttrDF, how='outer', left_on=['date', 'home_team_api_id'],
                            right_on=['date', 'team_api_id'])
    new_df_outer = pd.merge(new_df_outer, data_Team_AttrDF, how='outer', left_on=['date', 'away_team_api_id'],
                            right_on=['date', 'team_api_id'])

    new_df_outer = new_df_outer.sort_values(by=['home_team_api_id', 'away_team_api_id', 'date'])

    return new_df_outer




def addMeanGoalsLastSeasonByOther(data_df):

    """--------------------------------- New ------------------------------------"""

    data_home_df = data_df.groupby(['home_team_api_id', 'season'], as_index=False)['away_team_goal'].mean()
    data_away_df = data_df.groupby(['away_team_api_id', 'season'], as_index=False)['home_team_goal'].mean()

    copy_data_home_df = data_home_df.copy()
    copy_data_away_df = data_away_df.copy()
    seasons = pd.DataFrame(data_home_df['season'])
    seasons = seasons.groupby(['season'])

    new_seasons = []
    old_seasons = []
    i = 0
    for x in seasons:
        if i == 0:
            i += 1
            old_seasons.append(x[0])
            continue
        new_seasons.append(x[0])
        old_seasons.append(x[0])
    old_seasons.pop()

    copy_data_home_df = copy_data_home_df[copy_data_home_df.season != '2015/2016']
    copy_data_away_df = copy_data_away_df[copy_data_away_df.season != '2015/2016']

    copy_data_home_df = copy_data_home_df.replace(old_seasons, new_seasons)
    copy_data_away_df = copy_data_away_df.replace(old_seasons, new_seasons)

    data_df = pd.merge(data_df, copy_data_home_df, how='left', left_on=['home_team_api_id', 'season'], right_on=['home_team_api_id', 'season'])
    data_df = pd.merge(data_df, copy_data_away_df, how='left', left_on=['away_team_api_id', 'season'], right_on=['away_team_api_id', 'season'])

    # data_df['goals_mean'] = np.floor(data_df['home_team_goal_y'] / data_df['away_team_goal_y'])
    # del data_df['home_team_goal_y']
    # del data_df['away_team_goal_y']

    data_df.rename(columns={'home_team_goal_x': 'home_team_goal'}, inplace=True)
    data_df.rename(columns={'away_team_goal_x': 'away_team_goal'}, inplace=True)

    data_df.rename(columns={'home_team_goal_y': 'home_other_season_team_goal'}, inplace=True)
    data_df.rename(columns={'away_team_goal_y': 'away_other_season_team_goal'}, inplace=True)

    # print(data_df.apply(lambda x: sum(x.isnull()), axis=0))
    data_df = data_df.dropna()
    # print(data_df.apply(lambda x: sum(x.isnull()), axis=0))





    """--------------------------------- Origin ------------------------------------"""


    # data_home_df = data_df.groupby(['home_team_api_id', 'season'], as_index=False)['away_team_goal'].mean()
    # data_away_df = data_df.groupby(['away_team_api_id', 'season'], as_index=False)['home_team_goal'].mean()
    #
    # data_df = pd.merge(data_df, data_home_df, how='left', left_on=['home_team_api_id', 'season'], right_on=['home_team_api_id', 'season'])
    # data_df = pd.merge(data_df, data_away_df, how='left', left_on=['away_team_api_id', 'season'], right_on=['away_team_api_id', 'season'])
    #
    # # data_df['goals_mean'] = np.floor(data_df['home_team_goal_y'] / data_df['away_team_goal_y'])
    # # del data_df['home_team_goal_y']
    # # del data_df['away_team_goal_y']
    #
    # data_df.rename(columns={'home_team_goal_x': 'home_team_goal'}, inplace=True)
    # data_df.rename(columns={'away_team_goal_x': 'away_team_goal'}, inplace=True)
    #
    # data_df.rename(columns={'home_team_goal_y': 'home_other_season_team_goal'}, inplace=True)
    # data_df.rename(columns={'away_team_goal_y': 'away_other_season_team_goal'}, inplace=True)

    """------------------------------------------------------------------------------"""


    return data_df





