import pandas as pd
import matplotlib as plt
import numpy as np
from matplotlib.pyplot import show
import shap

def binning(col, cut_points, labels=None):
    # Define min and max values:
    minval = col.min()
    maxval = col.max()
    # create list by adding min and max to cut_points
    break_points = [minval] + cut_points + [maxval]
    # if no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points)+1)
    # Binning using cut function of pandas
    colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)
    return colBin



def check_data():
    # Reading the dataset in a dataframe using Pandas
    df = pd.read_csv("2012_2013_2014.csv")

    # print the data types of the attributes in the DataFrame
    print(df.dtypes)

    # Frequency distribution for non-numerical attributes
    for col in df:
        print(df[col].value_counts())

    print(df.describe())
    print(df.apply(lambda x: sum(x.isnull()), axis=0))

    df['home_team_goal'].hist()
    show()
    df['away_team_goal'].hist()
    show()

    temp3 = pd.crosstab(df['stage'], df['result'])
    temp3.plot(kind='bar', stacked=True, color=['red', 'blue', 'green'], grid=False)
    show()


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

    # bins = [1, 2, 3]
    # group_names = ['0', '1', '2', '3+']
    # # Discretize the values in LoanAmount attribute
    # df["home_team_goal_Bin"] = binning(df["home_team_goal"], bins, group_names)
    # # Count the number of observations which each value
    # print (pd.value_counts(df["home_team_goal_Bin"], sort=False))
    # print(df)
    df['home_team_goal_bin'].hist()
    show()
    df['away_team_goal_bin'].hist()
    show()


