import random

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler



# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


dic_of_team_id = {}


def model_KNN(trainData, testData):


    trainData = trainData.loc[:, ~trainData.columns.duplicated()]

    del trainData["home_team_api_id"]
    del trainData["away_team_api_id"]

    trainData = convertFeaturesToNumeric(trainData.copy())

    ####################################################################
    # Moshe Convert Values
    # trainData = DataFrame_Info_String2Numeric(trainData.copy())
    ####################################################################

    X = trainData.iloc[:, :-1].values
    y = trainData.iloc[:, len(trainData.columns) - 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    classifier = KNeighborsClassifier(n_neighbors=13)
    classifier.fit(X_train, y_train)

    y_predict = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_predict))
    print(classification_report(y_test, y_predict))
    print("\n**************************\n")
    print("Training data accuracy is", str(accuracy_score(y_test, y_predict)), "%")
    print("\n**************************\n")


    #########################################

    testData = testData.loc[:, ~testData.columns.duplicated()]

    del testData["home_team_api_id"]
    del testData["away_team_api_id"]

    testData = convertFeaturesToNumeric(testData.copy())

    ####################################################################
    # Moshe Convert Values
    # testData = DataFrame_Info_String2Numeric(testData.copy())
    ####################################################################


    season_15_16_features = testData.iloc[:, :-1].values
    season_15_16_Result = testData.iloc[:, len(testData.columns) - 1].values

    predict_15_16 = classifier.predict(season_15_16_features)

    print("\n**************************\n")
    print("Seasons 2015/2016 data accuracy is", str(accuracy_score(season_15_16_Result, predict_15_16)), "%")
    print("\n**************************\n")



def convertFeaturesToNumeric(dataToConvert):
    for col in dataToConvert:
        if col == "home_team_api_id":
            # dataToConvert[col] = dataToConvert[col].apply(changeTeamID)
            continue
        if col == "away_team_api_id":
            # dataToConvert[col] = dataToConvert[col].apply(changeTeamID)
            continue
        dataToConvert[col] = dataToConvert[col].apply(helpFuncForConvert)

    return dataToConvert


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


def changeTeamID(x):
    if str(x) not in dic_of_team_id:
        dic_of_team_id[str(x)] = random.uniform(0, 3)
    return dic_of_team_id[str(x)]


def DataFrame_Info_String2Numeric(data):
    le = preprocessing.LabelEncoder()
    for col in data.columns:
        if isinstance(data[col][0], str) and "name" not in col:
            # turn a string label into a number
            data[col] = le.fit_transform(data[col])
    return data


