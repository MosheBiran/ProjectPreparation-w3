import sqlite3
from aifc import Error


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


def select_task_by_Table(conn, Table):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param Table:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM " + Table)

    rows = cur.fetchall()

    for row in rows:
        print(row)


def main():
    database = r"C:\Users\biran\Desktop\3\database.sqlite\database.sqlite"

    # create a database connection
    conn = create_connection(database)
    with conn:
        print("1. Query by Table:")
        select_task_by_Table(conn, "Country")


if __name__ == '__main__':
    main()
