import pandas as pd
import matplotlib as plt
import numpy as np
from matplotlib.pyplot import show


def binGoals(df):
    """
    Binning The Goals Number Into Better Distribution
    :param df: The DataFrame That Contain The Goals Before The Binning
    :return: The DataFrame After The Binning
    """
    # Adding a column of discretization to the home goals.
    conditions = [df["home_team_goal"] == 0,
                  df["home_team_goal"] == 1,
                  df["home_team_goal"] == 2,
                  df["home_team_goal"] >= 3]

    choices = ["0", "1", "2", "3+"]
    df["home_team_goal_bin"] = np.select(conditions, choices, default=np.nan)

    # Adding a column of discretization to the away goals.
    conditions = [df["away_team_goal"] == 0,
                  df["away_team_goal"] == 1,
                  df["away_team_goal"] == 2,
                  df["away_team_goal"] >= 3]

    choices = ["0", "1", "2", "3+"]
    df["away_team_goal_bin"] = np.select(conditions, choices, default=np.nan)

    return df


def check_data(df):
    # print the data types of the attributes in the DataFrame
    print(df.dtypes)
    print(df.apply(lambda x: sum(x.isnull()), axis=0))

    # Frequency distribution for non-numerical attributes
    for col in df:
        print(df[col].value_counts())
        df[col].hist()
        show()

        temp3 = pd.crosstab(df[col], df['Result'])
        temp3.plot(kind='bar', stacked=True, color=['red', 'blue', 'green'], grid=False)
        show()




    print(df.describe())
    # print(df.apply(lambda x: sum(x.isnull()), axis=0))

    # df['home_team_goal'].hist()
    # show()
    # df['away_team_goal'].hist()
    # show()
    #
    # temp3 = pd.crosstab(df['stage'], df['result'])
    # temp3.plot(kind='bar', stacked=True, color=['red', 'blue', 'green'], grid=False)
    # show()

    # bins = [1, 2, 3]
    # group_names = ['0', '1', '2', '3+']
    # # Discretize the values in LoanAmount attribute
    # df["home_team_goal_Bin"] = binning(df["home_team_goal"], bins, group_names)
    # # Count the number of observations which each value
    # print (pd.value_counts(df["home_team_goal_Bin"], sort=False))
    # print(df)
    # df['home_team_goal_bin'].hist()
    # show()
    # df['away_team_goal_bin'].hist()
    # show()
