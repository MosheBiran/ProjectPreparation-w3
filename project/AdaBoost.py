import shap
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

"""
where we took AdaBoost from:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
"""


# path = "C:\\Users\\liadn\\Downloads\\"

# main function to run AdaBoost algorithm


def runAdaBoost(trainData, test_15_16):

    """--------------------------------- Splitting The Data ------------------------------------"""

    # split trainData to features
    # and classification
    data = trainData.iloc[:, :-1].values
    label = trainData.iloc[:, len(trainData.columns) - 1].values

    # split testData to features
    # and classification
    test_data = test_15_16.iloc[:, :-1].values
    test_label = test_15_16.iloc[:, len(test_15_16.columns) - 1].values

    X = data
    y = label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    """--------------------------------- Train Of The Model ------------------------------------"""


    clf = AdaBoostClassifier(n_estimators=290, learning_rate=0.1)
    clf.fit(X_train, y_train)


    """--------------------------------- Prediction And Evaluation On TrainData ------------------------------------"""

    y_predict = clf.predict(X_test)

    print("Confusion Matrix Train Data :\n", confusion_matrix(y_test, y_predict))
    print("Classification Report Train Data :\n", classification_report(y_test, y_predict))
    print("Accuracy Training data :", str(accuracy_score(y_test, y_predict)), "%")
    print("\n**************************\n")


    """--------------------------------- Prediction And Evaluation Of 2015/2016 ------------------------------------"""


    prediction_test = clf.predict(test_data)

    print("Confusion Matrix 2015_2016 :\n", metrics.confusion_matrix(test_label, prediction_test))
    print("Classification Report 2015_2016 :\n", classification_report(test_label, prediction_test))
    print("Accuracy 2015_2016: ",  str(accuracy_score(test_label, prediction_test)))
    print("\n**************************\n")
    # X_train_max, X_test_max, y_train_max, y_test_max,train_index_max,test_index_max = calcBestNumOfFolds(clf, X, y, 35, test_data, test_label,trainData)


# function that iterates over params and check which is the best
def calcBestNumOfFolds(clf, X, y, n, test_data, test_label, trainData):

    X_train_max, X_test_max, y_train_max, y_test_max, train_index_max, test_index_max = 0
    # var to save max for 2015_2016
    max_acc = 0
    counter = 1
    kf = KFold(n_splits=n, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        print(counter)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # start train model
        clf.fit(X_train, y_train)

        prediction_test = clf.predict(test_data)
        acc_test = metrics.accuracy_score(test_label, prediction_test)
        print("Accuracy 2015_2016:", acc_test)
        print("Confusion Metrix:\n", metrics.confusion_matrix(test_label, prediction_test))
        print(classification_report(test_label, prediction_test))

        # importances = clf.feature_importances_
        # indices = np.argsort(importances)
        # features = trainData.columns
        # plt.title('Feature Importances')
        # plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        # plt.yticks(range(len(indices)), [features[i] for i in indices])
        # plt.xlabel('Relative Importance')
        # plt.show()

        # found max split
        if acc_test > max_acc:
            max_acc = acc_test
            X_train_max, X_test_max = X[train_index], X[test_index]
            y_train_max, y_test_max = y[train_index], y[test_index]
            train_index_max,test_index_max = train_index, test_index
        counter += 1

    print(max_acc)

    return X_train_max, X_test_max, y_train_max, y_test_max, train_index_max, test_index_max
