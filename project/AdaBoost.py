import random
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
"""
where we took AdaBoost from:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
"""

# dictionary to help us select uniformly values
# that are normalized to team id
dic_of_team_id = {}

path = "C:\\Users\\liadn\\Downloads\\"
# main function to run AdaBoost algorithm
def runAdaBoost(trainData, test_15_16):

    # remove columns that are duplicated
    trainData = trainData.loc[:, ~trainData.columns.duplicated()]
    test_15_16 = test_15_16.loc[:, ~test_15_16.columns.duplicated()]

    # make our data applicable to the algorithm
    convertFeaturesToNumeric(trainData)
    convertFeaturesToNumeric(test_15_16)

    # helps us see rows that are
    # the same as others
    x1 = trainData.groupby(trainData.columns.tolist(), as_index=False).size()

    # split trainData to features
    # and classification
    data = trainData.iloc[:, :-1].values
    label = trainData.iloc[:, len(trainData.columns) - 1].values

    # split testData to features
    # and classification
    test_data = test_15_16.iloc[:, :-1].values
    test_label = test_15_16.iloc[:, len(test_15_16.columns) - 1].values

    # define Ada hyper params
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)


    X = data
    y = label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    # start train model
    clf.fit(X_train, y_train)


    # prediction of trained model
    y_pred = clf.predict(X_test)

    # calculate loss function
    loss = my_custom_loss_func(y_test, y_pred)

    # calculate confusion matrix
    arr = confusion_matrix(y_test, y_pred)
    print(arr)
    print("**********")

    # calculate model accuracy
    acc = accuracy_score(y_test, y_pred)
    print(acc)



def my_custom_loss_func(y_true, y_pred):
    diff = np.abs(y_true - y_pred).max()
    return np.log1p(diff)

def convertFeaturesToNumeric(dataToConvert):
    for col in dataToConvert:
        if col == "home_team_api_id":
            dataToConvert[col] = dataToConvert[col].apply(changeTeamID)
            continue
        if col == "away_team_api_id":
            dataToConvert[col] = dataToConvert[col].apply(changeTeamID)
            continue
        dataToConvert[col] = dataToConvert[col].apply(helpFuncForConvert)
    return dataToConvert

# created to help with converting not numeric values
# to numeric
def helpFuncForConvert(x):
    speedClass = ["Slow", "Balanced", "Fast"]
    dribblingClass = ["Little", "Normal", "Lots"]
    passingClass = ["Short", "Mixed", "Long"]
    positioningClass = ["Organised", "Free Form"]
    pressureClass = ["Medium", "Deep", "High"]
    aggressionClass = ["Press", "Double", "Contain"]
    goalClass = ["0", "1", "2", "3+"]
    whereBetterClass = ["Home", "Away", "NeverMind"]
    resultClass = ["Win", "Lose", "Draw"]

    if x in speedClass:
        return speedClass.index(x)
    elif x in dribblingClass:
        return dribblingClass.index(x)
    elif x in passingClass:
        return passingClass.index(x)
    elif x in positioningClass:
        return positioningClass.index(x)
    elif x in pressureClass:
        return pressureClass.index(x)
    elif x in aggressionClass:
        return aggressionClass.index(x)
    elif x in goalClass:
        return goalClass.index(x)
    elif x in whereBetterClass:
        return whereBetterClass.index(x)
    elif x in resultClass:
        return resultClass.index(x)

# in order to have un-biased values of team id
# we have uniformly distributed values of team id
def changeTeamID(x):
    if str(x) not in dic_of_team_id:
        dic_of_team_id[str(x)] = random.uniform(0, 3)
    return dic_of_team_id[str(x)]
