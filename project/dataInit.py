import sqlite3
from aifc import Error
import numpy as np
import pandas as pd
from scipy.interpolate import rbf
from sklearn.model_selection import train_test_split
from sklearn import svm


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


def save2CVS(database_after_clean, file_path):
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

def dataframe_filter_players(data_match_players_df, data_player_attr_df ):

    # Clearing the date from day and month
    data_match_players_df['date'] = data_match_players_df['date'].str.slice(stop=4)
    data_player_attr_df['date'] = data_player_attr_df['date'].str.slice(stop=4)
    data_player_attr_df = data_player_attr_df.groupby(['player_api_id', 'date']).mean()
    listt = data_player_attr_df.index.values
    id =[]
    years = []
    for val in listt:
        id.append(val[0])
        years.append(val[1])
    data_player_attr_df['date'] = years
    data_player_attr_df['player_api_id'] = id
    data_player_attr_df['rating'] = data_player_attr_df['overall_rating']
    del data_player_attr_df['overall_rating']
    data_player_attr_df.columns.name = ''

    data_match_players_df['sum'] = 0
    data_match_players_df['count'] = 0



    data_match_players_df = data_match_players_df.dropna()
    new_df_outer = pd.merge(data_match_players_df, data_player_attr_df, how='inner', left_on=['date', 'home_player_1'],
                            right_on=['date', 'player_api_id'])
    data_match_players_df['sum'] = data_match_players_df['sum'] +data_match_players_df['overall_rating']
    for i in range(1, 12):
        new_df_outer = pd.merge(data_match_players_df, data_player_attr_df, how='inner', left_on=['date', 'home_player_'+str(i)],
                            right_on=['date', 'team_api_id'])




    for index_match, row in data_match_players_df.iterrows():
        for index, rating in data_player_attr_df.iterrows():
            if row['date'] == index[1]:
                for i in range(1,12):
                    if row['home_player_' + str(i)] == index[0]:
                        row['sum'] += rating['overall_rating']
                        row['count'] += 1
                for i in range(1,12):
                    if row['away_player_' + str(i)] == index[0]:
                        row['sum'] += rating['overall_rating']
                        row['count'] += 1

    return data_match_players_df




def init():
    database = "database.sqlite"
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

    data_Players_AttrDF = pd.read_sql_query(
        'SELECT player_api_id,date,overall_rating from Player_Attributes',
        conn)

    data_matchDF_players = pd.read_sql_query(
        'SELECT home_team_api_id,away_team_api_id,season,date, home_player_1, home_player_2, home_player_3, '
        'home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, '
        'home_player_9, home_player_10, home_player_11 , away_player_1, away_player_2, away_player_3, away_player_4, '
        'away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11 '
        'from '
        'Match', conn)





    new_player_df = dataframe_filter_players(data_matchDF_players, data_Players_AttrDF)
    new_df = dataframe_filter(data_matchDF, data_Team_AttrDF)





    # Adding a column of binary representation win loss and draw.
    conditions = [new_df["home_team_goal"] > new_df["away_team_goal"],
                  new_df["home_team_goal"] < new_df["away_team_goal"],
                  new_df["home_team_goal"] == new_df["away_team_goal"]]

    choices = ["1", "-1", "0"]
    new_df["result"] = np.select(conditions, choices, default=np.nan)

    df_2012_2013_2014 = new_df.loc[(new_df['season'].isin(["2012/2013", "2013/2014", "2014/2015"]))]
    df_2012_2013_2014.to_csv('2012_2013_2014.csv')
    df_2015_2016 = new_df.loc[(new_df['season'].isin(["2015/2016"]))]

    # save2CVS(new_df, path)
    cursor.close()
    conn.close()
