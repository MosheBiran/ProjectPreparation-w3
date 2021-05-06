import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from matplotlib.pyplot import show
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


def model_SVM(trainData, testData):

    """--------------------------------- Splitting The Data ------------------------------------"""


    X = trainData.iloc[:, :-1].values
    y = trainData.iloc[:, len(trainData.columns) - 1].values

    X_T = testData.iloc[:, :-1].values
    y_T = testData.iloc[:, len(testData.columns) - 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    """--------------------------------- Train Of The Model ------------------------------------"""


    # making the instance
    # model = svm.SVC(kernel='linear', C=1000)
    model = svm.SVC(kernel='linear', C=12)
    # learning
    model.fit(X_train, y_train)


    """--------------------------------- Prediction And Evaluation On TrainData ------------------------------------"""


    # Prediction
    prediction = model.predict(X_test)

    # evaluation(Confusion Matrix)
    print("Confusion Matrix Train Data :\n", metrics.confusion_matrix(y_test, prediction))
    print("Classification Report Train Data :\n", classification_report(y_test, prediction))
    print("Accuracy Train Data :", str(accuracy_score(y_test, prediction)))
    print("\n**************************\n")


    """--------------------------------- Prediction And Evaluation Of 2015/2016 ------------------------------------"""

    prediction_test = model.predict(X_T)

    print("Confusion Matrix 2015_2016 :\n", confusion_matrix(y_T, prediction_test))
    print("Classification Report 2015_2016 :\n", classification_report(y_T, prediction_test))
    print("Accuracy 2015_2016 :", str(accuracy_score(y_T, prediction_test)))
    print("\n**************************\n")


    # print(classification_report(y_test, prediction))
    # print("\n**************************\n")
    # print("Training data accuracy is", str(accuracy_score(y_test, prediction)), "%")
    # print("\n**************************\n")

    # # The SHAP values
    # svm_explainer = shap.KernelExplainer(model.predict, X_T)
    # svm_shap_values = svm_explainer.shap_values(X_T)
    #
    # # shap.summary_plot(svm_shap_values, X_T)
    # # show()
    #
    # for col in testData:
    #     shap.dependence_plot(col, svm_shap_values, X_T)
    #     show()


# function that iterates over params and check which is the best
def calcBestNumOfFolds(clf, X, y, n, test_data, test_label, trainData):
    # var to save max for 2015_2016
    max_acc = 0
    """--------------------------------- Feature Scaling ------------------------------------"""
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    test_data = scaler.transform(test_data)

    X_train_max, X_test_max, y_train_max, y_test_max = 0

    kf = KFold(n_splits=n, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # start train model
        clf.fit(X_train, y_train)

        y_pred_test = clf.predict(test_data)
        acc_test = accuracy_score(test_label, y_pred_test)
        # found max split
        if acc_test > max_acc:
            max_acc = acc_test
            X_train_max, X_test_max = X[train_index], X[test_index]
            y_train_max, y_test_max = y[train_index], y[test_index]

    print(max_acc)
    return X_train_max, X_test_max, y_train_max, y_test_max
