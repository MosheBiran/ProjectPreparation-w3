import pandas as pd
import pip
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

"""
where we took AdaBoost from:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
"""

# path = "C:\\Users\\liadn\\Downloads\\"

# main function to run AdaBoost algorithm
def runAdaBoost(trainData, test_15_16):
    # trainData = trainData.drop(['chanceCreationPassing', 'chanceCreationCrossing', 'chanceCreationShooting', 'defenceTeamWidth'], axis=1)
    #
    # test_15_16 = test_15_16.drop(['chanceCreationPassing', 'chanceCreationCrossing', 'chanceCreationShooting', 'defenceTeamWidth'], axis=1)


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


    n_trees = [100]
    k_fold = [i for i in range(2,50)]
    for n in n_trees:
        for k in k_fold:
            print(k)
            clf = AdaBoostClassifier(n_estimators=n, random_state=0)
            # calcBestNumOfFolds(clf, X, y, 32, test_data, test_label)
            calcBestNumOfFolds(clf, X, y, 32, test_data, test_label,trainData)


#######################################
    # calcBestNumOfFolds(clf,X,y,5,test_data,test_label)
#######################################

    # # start train model
    # clf.fit(X_train, y_train)
    #
    # # prediction of trained model
    # y_pred = clf.predict(X_test)
    #
    # # prediction of trained model - *test*
    # y_pred_test = clf.predict(test_data)
    #
    # # calculate loss function
    # loss = my_custom_loss_func(y_test, y_pred)
    #
    # # calculate confusion matrix
    # arr = confusion_matrix(y_test, y_pred)
    # print(arr)
    # print("*****RUN ON 12_13_14*****")
    # # calculate model accuracy
    # acc = accuracy_score(y_test, y_pred)
    # print(acc)
    # print("*****RUN ON 15_16*****")
    # # calculate model accuracy
    # acc_test = accuracy_score(test_label, y_pred_test)
    # print(acc_test)



def my_custom_loss_func(y_true, y_pred):
    diff = np.abs(y_true - y_pred).max()
    return np.log1p(diff)


# function that iterates over params and check which is the best
def calcBestNumOfFolds(clf,X,y,n,test_data,test_label,trainData):
    # var to save max for 2015_2016
    max_acc = 0


    # for i in range(2,n):
        # max acc variable
    kf = KFold(n_splits=n, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # start train model
        clf.fit(X_train, y_train)

        importances = clf.feature_importances_
        indices = np.argsort(importances)
        features = trainData.columns
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

        y_pred_test = clf.predict(test_data)
        acc_test = accuracy_score(test_label, y_pred_test)
        # found max split
        if acc_test > max_acc:
            max_acc = acc_test
            X_train_max, X_test_max = X[train_index], X[test_index]
            y_train_max, y_test_max = y[train_index], y[test_index]
        break

    print(max_acc)


    # return X_train_max_for_tests, X_test_max_for_tests,y_train_max_for_tests, y_test_max_for_tests
