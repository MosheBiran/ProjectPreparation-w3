import sqlite3
from aifc import Error
import numpy as np

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
    database = r"C:\STUDY\YEAR C\SEMESTER B\סדנת הכנה לפרויקט\עבודות\3\archive\database.sqlite"
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
    data_match_np = np.array(data_match)

    cursor.execute('SELECT team_api_id,date,buildUpPlaySpeedClass,buildUpPlayDribblingClass,buildUpPlayPassingClass,buildUpPlayPositioningClass,defencePressureClass,defenceAggressionClass from Team_Attributes')
    data_Team_Attr = cursor.fetchall()
    data_Team_Attr_np = np.array(data_Team_Attr)

    # result_table = cursor.execute('''create table data (field1 real, field2 real, field3 real,    field4 real)''')
    # cursor.commit()

    res = []

    for i in range(0,len(data_match)):
        for j in range(0,len(data_Team_Attr)):
            # if year's date is the same
            if data_match[i][4][:3] != '2015' and data_match[i][4][:3] != '2016' and data_match[i][4][:3] != '2014' and data_match[i][4][:3] != '2013' and data_match[i][4][:3] != '2012':
                if data_match[i][4][:4] == data_Team_Attr[j][1][:4] and (data_match[i][0] == data_Team_Attr[j][0] or data_match[i][1] == data_Team_Attr[j][0]):
                    if len(res) != 0:
                        for k in range(0,len(res)):
                            if res[k][0] == data_Team_Attr[j][0] or res[k][1] == data_Team_Attr[j][0]:
                                a = list(res[k])
                                b = list(data_Team_Attr[j])
                                a.extend(b[2:])
                                res[k] = a
                        else:
                            a = list(data_match[i])
                            b = list(data_Team_Attr[j])
                            a.extend(b[2:])
                            res.append(a)
                    else:
                        a = list(data_match[i])
                        b = list(data_Team_Attr[j])
                        a.extend(b[2:])
                        res.append(a)

    cursor.close()
    conn.close()


