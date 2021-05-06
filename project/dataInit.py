import sqlite3
from aifc import Error

import numpy as np
import pandas as pd
from matplotlib.pyplot import show

from scipy.interpolate import rbf
from sklearn.model_selection import train_test_split
from sklearn import svm
from functools import reduce
import xml.etree.ElementTree as ET


from sklearn import preprocessing

path = "C:\\Users\\liadn\\Downloads\\"


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


def save2CSV(database_after_clean, file_path):
    """
    Save The Database As CSV File
    :param database_after_clean: The data that we want to save in CSV
    :param file_path: The Path where we want to save the database as CSV
    """
    database_after_clean.to_csv(file_path + "database_after_clean.csv")


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


def mergeMatchWithTeamAttribute_WithNull(data_match_df, data_team_attr_df):
    """
    Merging The Data Of:
    -The Matches With
    -The Data Attributes Of Each Team Playing in Each Match
    ** With Null **
    :param data_match_df: The Data about the matches
    :param data_team_attr_df: The Data about the team and there attributes
    :return: The Data merged with the matches and the attributes of the teams in the match - With Null
    """    # Clearing the date from day and month
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


# def dataframe_filter_players(data_match_players_df, player_attr_df,flag_pot,flag_shot_power,flag_gk,flag_free_kick):
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
    HomeAndAwayTeam_player_attr_mean_df['home_player_attr_mean'] = data_match_players_df[home_col_mean_lst].mean(
        1) / 100
    HomeAndAwayTeam_player_attr_mean_df['away_player_attr_mean'] = data_match_players_df[away_col_mean_lst].mean(
        1) / 100

    # HomeAndAwayTeam_player_attr_mean_df['players_rating'] = data_match_players_df[home_col_mean_lst].mean(1) / data_match_players_df[away_col_mean_lst].mean(1)

    """--------------------------------- All Other Features ------------------------------------"""
    """------------------ mixed ------------------"""
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
    # """------------------penalties-------------------"""
    # if list_of_fields[12] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'penalties_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'penalties_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_penalties_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_penalties_mean'] = data_match_players_df[away_col_mean_lst].mean(
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
    # """------------------agility-------------------"""
    # if list_of_fields[17] == True:
    #     home_col_mean_lst = [col for col in data_match_players_df.columns if 'agility_home_' in col]
    #     away_col_mean_lst = [col for col in data_match_players_df.columns if 'agility_away_' in col]
    #     HomeAndAwayTeam_player_attr_mean_df['home_player_agility_mean'] = data_match_players_df[home_col_mean_lst].mean(
    #         1) / 100
    #     HomeAndAwayTeam_player_attr_mean_df['away_player_agility_mean'] = data_match_players_df[away_col_mean_lst].mean(
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



def dataframe_attributeTeam_ratio(data_df):
    data_df['buildUpPlaySpeed'] = data_df['buildUpPlaySpeed_x']/data_df['buildUpPlaySpeed_y']
    del data_df['buildUpPlaySpeed_x']
    del data_df['buildUpPlaySpeed_y']

    data_df['buildUpPlayPassing'] = data_df['buildUpPlayPassing_x']/data_df['buildUpPlayPassing_y']
    del data_df['buildUpPlayPassing_x']
    del data_df['buildUpPlayPassing_y']

    data_df['chanceCreationPassing'] = data_df['chanceCreationPassing_x']/data_df['chanceCreationPassing_y']
    del data_df['chanceCreationPassing_x']
    del data_df['chanceCreationPassing_y']

    data_df['chanceCreationCrossing'] = data_df['chanceCreationCrossing_x']/data_df['chanceCreationCrossing_y']
    del data_df['chanceCreationCrossing_x']
    del data_df['chanceCreationCrossing_y']

    data_df['chanceCreationShooting'] = data_df['chanceCreationShooting_x']/data_df['chanceCreationShooting_y']
    del data_df['chanceCreationShooting_x']
    del data_df['chanceCreationShooting_y']

    data_df['defencePressure'] = data_df['defencePressure_x']/data_df['defencePressure_y']
    del data_df['defencePressure_x']
    del data_df['defencePressure_y']

    data_df['defenceAggression'] = data_df['defenceAggression_x']/data_df['defenceAggression_y']
    del data_df['defenceAggression_x']
    del data_df['defenceAggression_y']

    data_df['defenceTeamWidth'] = data_df['defenceTeamWidth_x']/data_df['defenceTeamWidth_y']
    del data_df['defenceTeamWidth_x']
    del data_df['defenceTeamWidth_y']

    return data_df


def dataframe_mean_goals(data_df):
    data_home_df = data_df.groupby(['home_team_api_id', 'season'], as_index=False)['home_team_goal'].mean()
    data_away_df = data_df.groupby(['away_team_api_id', 'season'], as_index=False)['away_team_goal'].mean()

    data_df = pd.merge(data_df, data_home_df, how='left', left_on=['home_team_api_id', 'season'], right_on=['home_team_api_id', 'season'])
    data_df = pd.merge(data_df, data_away_df, how='left', left_on=['away_team_api_id', 'season'], right_on=['away_team_api_id', 'season'])

    # data_df['goals_mean'] = np.floor(data_df['home_team_goal_y'] / data_df['away_team_goal_y'])
    # del data_df['home_team_goal_y']
    # del data_df['away_team_goal_y']

    data_df.rename(columns={'home_team_goal_x': 'home_team_goal'}, inplace=True)
    data_df.rename(columns={'away_team_goal_x': 'away_team_goal'}, inplace=True)

    data_df.rename(columns={'home_team_goal_y': 'home_season_team_goal'}, inplace=True)
    data_df.rename(columns={'away_team_goal_y': 'away_season_team_goal'}, inplace=True)



    # home_away_goals = pd.merge(data_home_df, data_away_df, how='outer', left_on=['home_team_api_id'], right_on=['away_team_api_id'])
    # home_away_goals['goal'] = home_away_goals[['home_team_goal','away_team_goal']].mean(1)
    # del home_away_goals['away_team_api_id']
    # del home_away_goals['home_team_goal']
    # del home_away_goals['away_team_goal']
    #
    # data_df = pd.merge(data_df, home_away_goals, how='left', left_on=['home_team_api_id'], right_on=['home_team_api_id'])
    # data_df = pd.merge(data_df, home_away_goals, how='left', left_on=['away_team_api_id'], right_on=['home_team_api_id'])
    # data_df['goals_mean'] = data_df['goal_x']/data_df['goal_y']
    #
    # data_df.rename(columns={'home_team_api_id_x': 'home_team_api_id'}, inplace=True)
    #
    # del data_df['goal_x']
    # del data_df['goal_y']
    # del data_df['home_team_api_id_y']

    return data_df



def dataframe_other_team_goals(data_df):

    data_home_df = data_df.groupby(['home_team_api_id', 'season'], as_index=False)['away_team_goal'].mean()
    data_away_df = data_df.groupby(['away_team_api_id', 'season'], as_index=False)['home_team_goal'].mean()

    data_df = pd.merge(data_df, data_home_df, how='left', left_on=['home_team_api_id', 'season'], right_on=['home_team_api_id', 'season'])
    data_df = pd.merge(data_df, data_away_df, how='left', left_on=['away_team_api_id', 'season'], right_on=['away_team_api_id', 'season'])

    # data_df['goals_mean'] = np.floor(data_df['home_team_goal_y'] / data_df['away_team_goal_y'])
    # del data_df['home_team_goal_y']
    # del data_df['away_team_goal_y']

    data_df.rename(columns={'home_team_goal_x': 'home_team_goal'}, inplace=True)
    data_df.rename(columns={'away_team_goal_x': 'away_team_goal'}, inplace=True)

    data_df.rename(columns={'home_team_goal_y': 'home_other_season_team_goal'}, inplace=True)
    data_df.rename(columns={'away_team_goal_y': 'away_other_season_team_goal'}, inplace=True)
    return data_df




def addTeamNames(new_df, data_team):
    """
    Add Into The Data The Teams Unique Names According The Team Api ID
    :param new_df: The Data that we want to add the teams names
    :param data_team: The Data with the teams names
    :return: The Final Data With The Teams Names Inside
    """
    new_df_with_name = pd.merge(data_team, new_df, how='inner', left_on=['team_api_id'],
                                right_on=['away_team_api_id'])

    new_df_with_name = pd.merge(data_team, new_df_with_name, how='inner', left_on=['team_api_id'],
                                right_on=['home_team_api_id'])

    del new_df_with_name['team_api_id_x']
    del new_df_with_name['team_api_id_y']
    new_df_with_name = remove_x_y(new_df_with_name)
    return new_df_with_name


def remove_x_y(new_df_with_name):
    """
    Removing From The Features Names the "_x" And "_y" -> "home_" And "away_"
    :param new_df_with_name: The Data that we want to convert
    :return: The Data After The Convert
    """
    for col in new_df_with_name.columns:
        if '_x' == col[len(col) - 2:len(col)]:
            new_df_with_name = new_df_with_name.rename(
                columns={col: 'home_' + col[:len(col) - 2]}
                , inplace=False)
        if '_y' == col[len(col) - 2:len(col)]:
            new_df_with_name = new_df_with_name.rename(
                columns={col: 'away_' + col[:len(col) - 2]}
                , inplace=False)

    return new_df_with_name


def sqlQuery(conn):
    """
    The SQL Queries For Getting The Data From The SQL Database
    :param conn: The Connection
    :return: The SQL Data As DataFrames - Match_DF, TeamAttributes_DF, Teams_DF
    """
    data_matchDF = pd.read_sql_query(
        'SELECT home_team_api_id,away_team_api_id,season,date,home_team_goal,away_team_goal from Match', conn)

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

def addingResultFeature(new_df):
    """
    Add Result Label For Each Game - According To The Home Team
    "Home Team Goals" - "Away Team Goals"
     2 = Win
     1 = Draw
     0 = Lose
    :param new_df: The Dataframe to Add The Label
    :return:The Data After Adding The Label "Result"
    """
    # Adding a column of binary representation win loss and draw.
    conditions = [new_df["home_team_goal"] > new_df["away_team_goal"],
                  new_df["home_team_goal"] < new_df["away_team_goal"],
                  new_df["home_team_goal"] == new_df["away_team_goal"]]
    # TODO : REMEBER CHOICES WAS [2,0,1]
    choices = [2, 0, 1]
    new_df["result"] = np.select(conditions, choices, default=np.nan)
    new_df["result"] = new_df["result"].astype(int)

    return new_df


def resultToCategorical(new_df):
    """
    Converting the result feature from int to categorical [2, 1, 0] -> ["Win", "Draw", "Lose"]
    :param new_df: The Dataframe that needed to convert
    :return: The Dataframe after the convert
    """
    # Adding a column of binary representation win loss and draw.
    conditions = [new_df["result"] == 2,
                  new_df["result"] == 1,
                  new_df["result"] == 0]

    choices = ["Win", "Draw", "Lose"]
    new_df["result"] = np.select(conditions, choices, default=np.nan)
    new_df["HomeTeamResult"] = new_df["result"]
    del new_df["result"]

    return new_df


def clearUnusedFeatures(new_df):
    """
    Cleaning The Features That We No Longer Needed
    After The Init And
    Before The Train
    :param new_df: The Dataframe that we need to clear
    :return: The Dataframe after clean
    """
    del new_df["home_team_goal"]
    del new_df["away_team_goal"]
    del new_df["season"]
    del new_df["date"]
    del new_df["home_team_api_id"]
    del new_df["away_team_api_id"]
    del new_df["home_percentHome"]
    del new_df["home_percentAway"]
    del new_df["away_percentHome"]
    del new_df["away_percentAway"]


    del new_df["away_whereBetter"]
    del new_df["home_whereBetter"]


    del new_df["home_season_team_goal"]
    del new_df["home_other_season_team_goal"]
    del new_df["away_season_team_goal"]
    del new_df["away_other_season_team_goal"]


    new_df["Result"] = new_df["result"]
    del new_df["result"]

    return new_df


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


def DataFrame_Info_String2Numeric(data):
    le = preprocessing.LabelEncoder()
    for col in data.columns:
        if isinstance(data[col][0], str) and "name" not in col:
            # turn a string label into a number
            data[col] = le.fit_transform(data[col])
    return data


def addLastMatchesGoals(match_Data_DF):

    # origin_DF = match_Data_DF.copy()
    # origin_DF = origin_DF.sort_values(by=['home_team_api_id', 'date'])  # TODO : No Need


    """--------------------------------- Splitting To Home And Away------------------------------------"""


    match_sorted_by_home = match_Data_DF.sort_values(by=['home_team_api_id', 'date'])
    match_sorted_by_home = match_sorted_by_home[['home_team_api_id', 'date', 'home_team_goal']]

    match_sorted_by_away = match_Data_DF.sort_values(by=['away_team_api_id', 'date'])
    match_sorted_by_away = match_sorted_by_away[['away_team_api_id', 'date', 'away_team_goal']]


    """--------------------------------- Home Team Goals Calculate ------------------------------------"""


    match_sorted_by_home.insert(len(match_sorted_by_home.columns), 'Home_LastMatchesGoals', match_sorted_by_home['home_team_goal'])
    inxOfCol = len(match_sorted_by_home.columns) - 1

    homeAfterAdd = homeOrAwayGoals(match_sorted_by_home, inxOfCol, 'home_')


    """--------------------------------- Away Team Goals Calculate ------------------------------------"""


    match_sorted_by_away.insert(len(match_sorted_by_away.columns), 'Away_LastMatchesGoals', match_sorted_by_away['away_team_goal'])
    inxOfCol = len(match_sorted_by_away.columns) - 1

    awayAfterAdd = homeOrAwayGoals(match_sorted_by_away, inxOfCol, 'away_')


    """--------------------------------- Merge To Main Data ------------------------------------"""


    match_Data_DF = pd.merge(match_Data_DF, homeAfterAdd, how='inner',
                             left_on=['home_team_api_id', 'date', 'home_team_goal'],
                             right_on=['home_team_api_id', 'date', 'home_team_goal'])

    match_Data_DF = pd.merge(match_Data_DF, awayAfterAdd, how='inner',
                             left_on=['away_team_api_id', 'date', 'away_team_goal'],
                             right_on=['away_team_api_id', 'date', 'away_team_goal'])


    # # TODO : For Check
    # match_Data_DF = match_Data_DF.sort_values(by=['home_team_api_id', 'date'])
    # match_Data_DF['Home_LastMatchesGoals'].hist()
    # print(match_Data_DF['Home_LastMatchesGoals'].value_counts())
    # show()
    # match_Data_DF['Away_LastMatchesGoals'].hist()
    # print(match_Data_DF['Away_LastMatchesGoals'].value_counts())
    # show()

    return match_Data_DF





def homeOrAwayGoals(sortedHomeOrAway, inxOfCol, home_away, gamesBack=5):
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
    match_Data_DF, team_Attr_Data_DF, teams_Data_DF, data_Players_AttrDF, data_matchDF_players = sqlQuery(conn)


    # TODO : Try Add 5 Past Games Goal
    match_Data_DF = addLastMatchesGoals(match_Data_DF)


    """--------------------------------- Merging All The DataFrames Into One ------------------------------------"""


    # Players_Attr_avg = dataframe_filter_players(data_matchDF_players, data_Players_AttrDF,flag_pot,flag_shot_power,flag_gk,flag_free_kick)
    Players_Attr_avg = dataframe_filter_players(data_matchDF_players, data_Players_AttrDF)

    matchWithTeamAttributes_df = mergeMatchWithTeamAttribute(match_Data_DF, team_Attr_Data_DF)

    """--------------------------------- Adding New Features ------------------------------------"""


    # Adding Label Result To The Data
    matchWithTeamAttributes_df = addingResultFeature(matchWithTeamAttributes_df)

    matchWithTeamAttributes_df = pd.merge(matchWithTeamAttributes_df, Players_Attr_avg, how='inner', left_on=['home_team_api_id', 'away_team_api_id', 'season', 'date'], right_on=['home_team_api_id', 'away_team_api_id', 'season', 'date'])
    matchWithTeamAttributes_df = dataframe_mean_goals(matchWithTeamAttributes_df)

    matchWithTeamAttributes_df = dataframe_attributeTeam_ratio(matchWithTeamAttributes_df)

    matchWithTeamAttributes_df = dataframe_other_team_goals(matchWithTeamAttributes_df)

    # Calculate Where The Team Playing Better
    trainData_before_WB = matchWithTeamAttributes_df.loc[(matchWithTeamAttributes_df['season'].isin(["2012/2013", "2013/2014", "2014/2015"]))]
    testData_before_WB = matchWithTeamAttributes_df.loc[(matchWithTeamAttributes_df['season'].isin(["2015/2016"]))]

    # Merging WhereBetter With Main Data
    trainData, testData = margeWhereBetterWithMainData(trainData_before_WB, testData_before_WB)

    #################################################################################################################################
    # # Convert Class Result To Categorical  # TODO: Not Need For Categorical
    # trainData = resultToCategorical(trainData)
    # testData = resultToCategorical(testData)
    #################################################################################################################################


    """--------------------------------- Clearing The UnNeeded Features And Converting To Numeric ------------------------------------"""


    trainData = clearUnusedFeatures(trainData)
    testData = clearUnusedFeatures(testData)

    trainData = DataFrame_Info_String2Numeric(trainData.copy())
    testData = DataFrame_Info_String2Numeric(testData.copy())

    cursor.close()
    conn.close()

    return trainData, testData



# def temp():
#     database = path + "database.sqlite"
#
#     # create a database connection
#     conn = create_connection(database)
#     cursor = conn.cursor()
#
#     data_matchDF = pd.read_sql_query(
#         'SELECT home_team_api_id,away_team_api_id, shoton from Match', conn)
#
#     print(data_matchDF.apply(lambda x: sum(x.isnull()), axis=0))
#
#     for x in data_matchDF['shoton']:
#         if x is None:
#             continue
#         root = ET.XML(x)  # Parse XML
#
#         data = []
#         cols = []
#         flag = 0
#         for i, child in enumerate(root.iter()):
#             print(child.tag)
#             if child.tag == 'shoton':
#                 flag += 1
#
#             if flag <= 1:
#                 cols.append(child.tag)
#
#             for subchild in child:
#                 data.append(subchild.text)
#
#         df = pd.DataFrame(data).T  # Write in DF and transpose it
#         df.columns = cols  # Update column names
#         print(df)
#
#     cursor.close()
#     conn.close()
