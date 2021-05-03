import sqlite3
from aifc import Error
from functools import reduce

import numpy as np
import pandas as pd
from matplotlib.pyplot import show
from scipy.interpolate import rbf
from sklearn.model_selection import train_test_split
from sklearn import svm

from project import preprocessing

path = "C:\\Users\\Daniel\\Downloads\\archive\\"



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
    """
    # Clearing the date from day and month
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
        'SELECT team_api_id,date,buildUpPlaySpeedClass,buildUpPlayDribblingClass,buildUpPlayPassingClass,'
        'buildUpPlayPositioningClass,defencePressureClass,defenceAggressionClass from Team_Attributes',
        conn)

    data_Team = pd.read_sql_query('SELECT team_api_id, team_long_name from Team', conn)

    return data_matchDF, data_Team_AttrDF, data_Team


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
    df_home_team_win_sum = df_home_team_win_sum.loc[(df_home_team_win_sum['result'] == 1)]

    df_away_team_win_sum = new_df_with_name.groupby(["away_team_api_id", "result"]).size().reset_index(
        name="wins_away_sum")
    df_away_team_win_sum = df_away_team_win_sum.loc[(df_away_team_win_sum['result'] == -1)]

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
     1 = Win
     0 = Draw
    -1 = Lose
    :param new_df: The Dataframe to Add The Label
    :return:The Data After Adding The Label "Result"
    """
    # Adding a column of binary representation win loss and draw.
    conditions = [new_df["home_team_goal"] > new_df["away_team_goal"],
                  new_df["home_team_goal"] < new_df["away_team_goal"],
                  new_df["home_team_goal"] == new_df["away_team_goal"]]

    choices = [1, -1, 0]
    new_df["result"] = np.select(conditions, choices, default=np.nan)
    new_df["result"] = new_df["result"].astype(int)

    return new_df


def resultToCategorical(new_df):
    """
    Converting the result feature from int to categorical [1, 0, -1] -> ["Win", "Draw", "Lose"]
    :param new_df: The Dataframe that needed to convert
    :return: The Dataframe after the convert
    """
    # Adding a column of binary representation win loss and draw.
    conditions = [new_df["result"] == 1,
                  new_df["result"] == 0,
                  new_df["result"] == -1]

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
    # del new_df["home_team_api_id"]
    # del new_df["away_team_api_id"]
    del new_df["home_percentHome"]
    del new_df["home_percentAway"]
    del new_df["away_percentHome"]
    del new_df["away_percentAway"]

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


def init():
    """
    The Init And Building The Data From The Model Training And Testing
    :return: The Train Data , Test Data
    """
    database = path + "database.sqlite"

    # create a database connection
    conn = create_connection(database)
    cursor = conn.cursor()

    # create DF
    match_Data_DF, team_Attr_Data_DF, teams_Data_DF = sqlQuery(conn)

    teams_Data_DF = teams_Data_DF.sort_values(by=['team_api_id'])

    matchWithTeamAttributes_df = mergeMatchWithTeamAttribute(match_Data_DF, team_Attr_Data_DF)

    # Adding Label Result To The Data
    matchWithTeamAttributes_df = addingResultFeature(matchWithTeamAttributes_df)

    # Binning - The Number Of Goals
    dataWithTeamNames = matchWithTeamAttributes_df = preprocessing.binGoals(matchWithTeamAttributes_df)

    # Adding The Names Of The Teams
    # dataWithTeamNames = addTeamNames(matchWithTeamAttributes_df, teams_Data_DF)

    # Calculate Where The Team Playing Better
    trainData_before_WB = dataWithTeamNames.loc[(dataWithTeamNames['season'].isin(["2012/2013", "2013/2014", "2014/2015"]))]
    testData_before_WB = dataWithTeamNames.loc[(dataWithTeamNames['season'].isin(["2015/2016"]))]

    # Merging WhereBetter With Main Data
    trainData, testData = margeWhereBetterWithMainData(trainData_before_WB, testData_before_WB)

    # Convert Class Result To Categorical
    trainData = resultToCategorical(trainData)
    testData = resultToCategorical(testData)

    trainData = clearUnusedFeatures(trainData)
    testData = clearUnusedFeatures(testData)

    cursor.close()
    conn.close()

    return trainData, testData
