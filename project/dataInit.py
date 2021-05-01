import sqlite3
from aifc import Error
import numpy as np
import pandas as pd


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


def init():
    path = "C:\\Users\\biran\\Desktop\\3\\database.sqlite\\"
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

    # Clearing the date from day and month
    data_matchDF['date'] = data_matchDF['date'].str.slice(stop=4)
    data_Team_AttrDF['date'] = data_Team_AttrDF['date'].str.slice(stop=4)

    # sorting by relevant col
    data_matchDF = data_matchDF.sort_values(by=['home_team_api_id', 'away_team_api_id', 'date'])
    data_Team_AttrDF = data_Team_AttrDF.sort_values(by=['team_api_id', 'date'])

    # merging first by ['date', 'home_team_api_id'] and again by ['date', 'away_team_api_id']
    inner_new_df = pd.merge(data_matchDF, data_Team_AttrDF, how='inner', left_on=['date', 'home_team_api_id'],
                            right_on=['date', 'team_api_id'])
    inner_new_df = pd.merge(inner_new_df, data_Team_AttrDF, how='inner', left_on=['date', 'away_team_api_id'],
                            right_on=['date', 'team_api_id'])

    # df with null
    null_new_df = pd.merge(data_matchDF, data_Team_AttrDF, how='outer', left_on=['date', 'home_team_api_id'],
                           right_on=['date', 'team_api_id'])
    null_new_df = pd.merge(null_new_df, data_Team_AttrDF, how='outer', left_on=['date', 'away_team_api_id'],
                           right_on=['date', 'team_api_id'])
    null_new_df = null_new_df.sort_values(by=['home_team_api_id', 'away_team_api_id', 'date'])

    # null_new_df['home_team_api_id'] = null_new_df['home_team_api_id'].astype(int)
    # null_new_df['away_team_api_id'] = null_new_df['away_team_api_id'].astype(int)

    # Adding a column of binary representation win loss and draw.
    conditions = [inner_new_df["home_team_goal"] > inner_new_df["away_team_goal"],
                  inner_new_df["home_team_goal"] < inner_new_df["away_team_goal"],
                  inner_new_df["home_team_goal"] == inner_new_df["away_team_goal"]]

    choices = ["1", "-1", "0"]
    inner_new_df["result"] = np.select(conditions, choices, default=np.nan)

    # temp=inner_new_df["season"][1]
    # inner_new_df.loc[
    #     (inner_new_df["season"] == 2012 / 2013) & inner_new_df["season"] == 2013 / 2014 & inner_new_df[
    #         "season"] == 2014 / 2015]
    # x=1
    # save2CVS(inner_new_df, path)
    cursor.close()
    conn.close()
