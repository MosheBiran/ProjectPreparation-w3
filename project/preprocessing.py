import pandas as pd


def check_data():
    # Reading the dataset in a dataframe using Pandas
    df = pd.read_csv("database_null.csv")

    # print the data types of the attributes in the DataFrame
    print(df.dtypes)

    # Frequency distribution for non-numerical attributes
    for col in df:
        print(df[col].value_counts())

    print(df.describe())
    print(df.apply(lambda x: sum(x.isnull()), axis=0))
