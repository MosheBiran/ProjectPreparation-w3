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


def init():
    database = path + "database.sqlite"
    """
    Country = { id , name }
    League = { id , country_id , name  }
    Match = { id , country_id , league_id ... 115 column }
    Player = { id , player_api_id , player_name , player_fifa_api_id , birthday , height , weight }
    Player_Attributes = { 42 col }
    Team = {id , team_api_id , team_fifa_api_id , team_long_name , team_short_name }
    Team_Attributes = { 25 col }
    """
    # create a database connection
    conn = create_connection(database)
    cursor = conn.cursor()

    # create DF
    data_matchDF = pd.read_sql_query(
        'SELECT home_team_api_id,away_team_api_id,season,stage,date,home_team_goal,away_team_goal from Match', conn)

    data_Team_AttrDF = pd.read_sql_query(
        'SELECT team_api_id,date,buildUpPlaySpeedClass,buildUpPlayDribblingClass,buildUpPlayPassingClass,'
        'buildUpPlayPositioningClass,defencePressureClass,defenceAggressionClass from Team_Attributes',
        conn)

    data_Team = pd.read_sql_query('SELECT team_api_id, team_long_name from Team', conn)
    data_Team = data_Team.sort_values(by=['team_api_id'])

    # data_Team = data_Team.dropna(thresh=2)
    # data_Team['team_api_id'] = data_Team['team_api_id'].astype(int)

    new_df = dataframe_filter(data_matchDF, data_Team_AttrDF)

    # Adding team long name
    new_df = pd.merge(data_Team, new_df, how='inner', left_on=['team_api_id'],
                      right_on=['away_team_api_id'])

    new_df = pd.merge(data_Team, new_df, how='inner', left_on=['team_api_id'],
                      right_on=['home_team_api_id'])

    del new_df['team_api_id_x']
    del new_df['team_api_id_y']

    new_df = new_df.rename(columns={'team_long_name_x': 'home_team_long_name','team_long_name_y': 'away_team_long_name'}
                           , inplace=False)

    # Adding a column of binary representation win loss and draw.
    conditions = [new_df["home_team_goal"] > new_df["away_team_goal"],
                  new_df["home_team_goal"] < new_df["away_team_goal"],
                  new_df["home_team_goal"] == new_df["away_team_goal"]]

    choices = ["1", "-1", "0"]
    new_df["result"] = np.select(conditions, choices, default=np.nan)

    test_train_models_split(new_df)
    # save2CSV(new_df, path)
    cursor.close()
    conn.close()
