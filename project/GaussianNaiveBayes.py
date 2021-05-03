from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix


def DataFrame_Info_String2Numeric(data):
    le = preprocessing.LabelEncoder()
    for col in data.columns:
        if isinstance(data[col][0], str) and "name" not in col:
            # turn a string label into a number
            data[col] = le.fit_transform(data[col])
    return data


def split_DataFrame_2_TrainAndTest(data):
    X_ = data.drop(["home_team_long_name", "away_team_long_name", "HomeTeamResult"], axis=1)
    y_ = data["HomeTeamResult"]
    return X_, y_


def print_ModelResult(y, model_prediction, test_name=''):
    print("---------------------------------" + str(test_name) + "---------------------------------------")
    print("Training data accuracy is", str(accuracy_score(y, model_prediction)), "%")
    print(confusion_matrix(y, model_prediction))
    print(classification_report(y, model_prediction))


def print_PredictionResult(test_data_numbers, future_predictions):
    for count in range(0, len(future_predictions)):
        if future_predictions[count] == 1:
            print(test_data_numbers["home_team_long_name"][count], "versus", test_data_numbers["away_team_long_name"][count], "is Home Win")
        elif future_predictions[count] == 2:
            print(test_data_numbers["home_team_long_name"][count], "versus", test_data_numbers["away_team_long_name"][count], "is Draw")
        else:
            print(test_data_numbers["home_team_long_name"][count], "versus", test_data_numbers["away_team_long_name"][count], "is Away Win")


def naive_bayes_function(train_data, test_data):

    model = GaussianNB()

    train_data_numbers = DataFrame_Info_String2Numeric(train_data.copy())
    test_data_numbers = DataFrame_Info_String2Numeric(test_data.copy())

    X_train_model, y_train_model = split_DataFrame_2_TrainAndTest(train_data_numbers)
    X_test_model, y_test_model = split_DataFrame_2_TrainAndTest(test_data_numbers)

    model.fit(X_train_model, y_train_model)
    training_predictions = model.predict(X_train_model)
    future_predictions = model.predict(X_test_model)

    print_ModelResult(y_train_model, training_predictions, "y_2train+X_2train")
    print_ModelResult(y_test_model, future_predictions, "y_2test+X_2test")

    # print("-----------------X_train, X_test, y_train, y_test----------------")
    # # print("Training data recall is", str(recall_score(y_train, training_predictions)), "%")
    # # print("Training data precision is", str(precision_score(y_train, training_predictions)), "%")
    #
    # # Create a Gaussian Classifier
    # X_train, X_test, y_train, y_test = train_test_split(X_2train, y_2train, test_size=0.2, random_state=0)
    # y_pred = model.fit(X_train, y_train).predict(X_test)
    # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    #
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))




    # zip together 2 cols
    # features = zip(weather_encoded, temp_encoded)
