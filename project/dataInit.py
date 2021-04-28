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


def init():
    database = r"C:\Users\Daniel\Downloads\archive\database.sqlite"
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

    cursor.execute('SELECT home_team_api_id,away_team_api_id,season,stage,date,home_team_goal,away_team_goal from Match')
    data_match = cursor.fetchall()
    # data_match_np = np.array(data_match)

    matches = pd.read_sql('SELECT home_team_api_id,away_team_api_id,season,stage,date,home_team_goal,away_team_goal from Match', conn)

    teams = pd.read_sql('SELECT team_api_id,date,buildUpPlaySpeedClass,buildUpPlayDribblingClass,buildUpPlayPassingClass,buildUpPlayPositioningClass,defencePressureClass,defenceAggressionClass from Team_Attributes', conn)


    merged = pd.merge(matches, teams, how='outer', left_on="home_team_api_id", right_on="team_api_id")
    


    # cursor.execute('SELECT team_api_id,date,buildUpPlaySpeedClass,buildUpPlayDribblingClass,buildUpPlayPassingClass,buildUpPlayPositioningClass,defencePressureClass,defenceAggressionClass from Team_Attributes')
    # data_Team_Attr = cursor.fetchall()
    #
    # data_Team_Attr_np = np.array(data_Team_Attr)
    # AllData = np.zeros((len(data_match_np), 27), dtype=str)
    #
    # for i in range(len(data_match_np)):
    #     for k in range(7):
    #         AllData[i][k] = str(data_match_np[i][k])
    #     for j in range(len(data_Team_Attr_np)):
    #         #  Home Team
    #         if data_match_np[i][0] == data_Team_Attr_np[j][0]:
    #             if data_match_np[i][4][:4] == data_Team_Attr_np[j][1][:4]:
    #                 for k in range(7, 7+len(data_Team_Attr_np[j])-2):
    #                     AllData[i][k] = str(data_Team_Attr_np[j][k-5])
    #         #  Away Team
    #         elif data_match_np[i][1] == data_Team_Attr_np[j][0]:
    #             if data_match_np[i][4][:4] == data_Team_Attr_np[j][1][:4]:
    #                 for k in range(7 + len(data_Team_Attr_np[j])-1,  7 + len(data_Team_Attr_np[j])-1 + len(data_Team_Attr_np[j])-2):
    #                     t=AllData[i][k] = str(data_Team_Attr_np[j][k - 12])
    #                     AllData[i][k] = str(data_Team_Attr_np[j][k - 12])
    #
    #         # TODO - its puts only the first Char..
    #         # TODO - in away team 1 step back all the data

    cursor.close()
    conn.close()


