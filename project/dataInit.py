import sqlite3
from aifc import Error
import numpy as np
import pandas as pd
from scipy.interpolate import rbf
from sklearn.model_selection import train_test_split
from sklearn import svm

path = "C:\\Users\\biran\\Desktop\\3\\database.sqlite\\"


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


def select_all_tasks(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT *")

    rows = cur.fetchall()

    for row in rows:
        print(row)


def select_task_by_table(conn, table):
    """
    Query tasks by table
    :param conn: the Connection object
    :param table:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM " + table)

    rows = cur.fetchall()

    for row in rows:
        print(row)


def create_table(cursor):
    cursor.execute('''create table data (field1 real, field2 real, field3 real,    field4 real)''')
    cursor.commit()


def save2CSV(database_after_clean, file_path):
    database_after_clean.to_csv(file_path + "database_after_clean.csv")


def dataframe_filter(data_match_df, data_team_attr_df):
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


def dataframe_filter_null(data_match_df, data_team_attr_df):
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


def test_train_models_split(new_df):
    df_2012_2013_2014 = new_df.loc[(new_df['season'].isin(["2012/2013", "2013/2014", "2014/2015"]))]
    df_2015_2016 = new_df.loc[(new_df['season'].isin(["2015/2016"]))]
    df_2012_2013_2014.to_csv(path + "df_2012_2013_2014.csv")
    df_2015_2016.to_csv(path + "df_2015_2016.csv")
    return df_2012_2013_2014, df_2015_2016


def get_team_names(new_df, data_team):
    new_df_with_name = pd.merge(data_team, new_df, how='inner', left_on=['team_api_id'],
                                right_on=['away_team_api_id'])

    new_df_with_name = pd.merge(data_team, new_df_with_name, how='inner', left_on=['team_api_id'],
                                right_on=['home_team_api_id'])

    del new_df_with_name['team_api_id_x']
    del new_df_with_name['team_api_id_y']

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


"""
Country = { id , name }
League = { id , country_id , name  }
Match = { id , country_id , league_id ... 115 column }
Player = { id , player_api_id , player_name , player_fifa_api_id , birthday , height , weight }
Player_Attributes = { 42 col }
Team = {id , team_api_id , team_fifa_api_id , team_long_name , team_short_name }
Team_Attributes = { 25 col }
"""


def sql_q(conn):
    data_matchDF = pd.read_sql_query(
        'SELECT home_team_api_id,away_team_api_id,season,stage,date,home_team_goal,away_team_goal from Match', conn)

    data_Team_AttrDF = pd.read_sql_query(
        'SELECT team_api_id,date,buildUpPlaySpeedClass,buildUpPlayDribblingClass,buildUpPlayPassingClass,'
        'buildUpPlayPositioningClass,defencePressureClass,defenceAggressionClass from Team_Attributes',
        conn)

    data_Team = pd.read_sql_query('SELECT team_api_id, team_long_name from Team', conn)

    return data_matchDF, data_Team_AttrDF, data_Team


def get_win_percent(new_df_with_name):
    df_home_team_win_sum = new_df_with_name.groupby(["home_team_api_id", "result"]).size().reset_index(name="wins_home_sum")
    df_home_team_win_sum = df_home_team_win_sum.loc[(df_home_team_win_sum['result'] == 1)]

    df_away_team_win_sum = new_df_with_name.groupby(["away_team_api_id", "result"]).size().reset_index(name="wins_away_sum")
    df_away_team_win_sum = df_away_team_win_sum.loc[(df_away_team_win_sum['result'] == -1)]

    df_home_team_total_count = new_df_with_name.groupby(["home_team_api_id"]).result.count().reset_index(
        name="home_count")

    df_away_team_total_count = new_df_with_name.groupby(["away_team_api_id"]).result.count().reset_index(
        name="away_count")

    del df_home_team_win_sum['result']
    del df_away_team_win_sum['result']

    df_home_team_win_sum = df_home_team_win_sum.merge(df_home_team_total_count, on=['home_team_api_id'], how='left')
    df_away_team_win_sum = df_away_team_win_sum.merge(df_away_team_total_count, on=['away_team_api_id'], how='left')

    df_away_team_win_sum['percentAway'] = df_away_team_win_sum[['wins_away_sum']].div(df_away_team_win_sum['away_count'], axis=0)
    df_home_team_win_sum['percentHome'] = df_home_team_win_sum[['wins_home_sum']].div(df_home_team_win_sum['home_count'], axis=0)

    df_percent_wim = pd.merge(df_home_team_win_sum, df_away_team_win_sum, how='inner', left_on=['home_team_api_id'],
                              right_on=['away_team_api_id'])

    conditions_percent = [df_percent_wim["percentHome"] > df_percent_wim["percentAway"],
                          df_percent_wim["percentHome"] < df_percent_wim["percentAway"],
                          df_percent_wim["percentHome"] == df_percent_wim["percentAway"]]

    choices = ["Home", "Away", "NeverMind"]
    df_percent_wim["whereBetter"] = np.select(conditions_percent, choices, default=np.nan)
    return df_percent_wim


def init():
    database = path + "database.sqlite"

    # create a database connection
    conn = create_connection(database)
    cursor = conn.cursor()

    # create DF
    data_matchDF, data_Team_AttrDF, data_Team = sql_q(conn)

    data_Team = data_Team.sort_values(by=['team_api_id'])

    new_df = dataframe_filter(data_matchDF, data_Team_AttrDF)

    # Adding a column of binary representation win loss and draw.
    conditions = [new_df["home_team_goal"] > new_df["away_team_goal"],
                  new_df["home_team_goal"] < new_df["away_team_goal"],
                  new_df["home_team_goal"] == new_df["away_team_goal"]]

    choices = [1, -1, 0]
    new_df["result"] = np.select(conditions, choices, default=np.nan)
    new_df["result"] = new_df["result"].astype(int)

    new_df_with_name = get_team_names(new_df, data_Team)
    df_2012_2013_2014 = new_df.loc[(new_df_with_name['season'].isin(["2012/2013", "2013/2014", "2014/2015"]))]
    df_percent_wim = get_win_percent(df_2012_2013_2014)

    # test_train_models_split(new_df_with_name)

    # save2CSV(new_df, path)
    cursor.close()
    conn.close()
