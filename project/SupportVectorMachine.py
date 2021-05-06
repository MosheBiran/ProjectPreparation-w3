import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from matplotlib.pyplot import show
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def model_SVM(trainData, testData):
    X = trainData.iloc[:, :-1].values
    y = trainData.iloc[:, len(trainData.columns) - 1].values

    X_T = testData.iloc[:, :-1].values
    y_T = testData.iloc[:, len(testData.columns) - 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    #making the instance
    # model = svm.LinearSVC()
    model = svm.SVC(kernel="linear",C=1000)
    # # # learning
    model.fit(X_train, y_train)
    # # # Prediction
    prediction_test = model.predict(X_T)
    # evaluation(Accuracy)
    print("Accuracy 2015_2016:", metrics.accuracy_score(y_T, prediction_test))
    # evaluation(Confusion Metrix)
    print("Confusion Metrix:\n", metrics.confusion_matrix(y_T, prediction_test))
    print(classification_report(y_T, prediction_test))





    # making the instance-K-Fold
    # model = svm.SVC(kernel="linear", C=1000)
    # Using K-Fold for best split of data and validation sets
    # X_train_max, X_test_max, y_train_max, y_test_max = calcBestNumOfFolds(model, X, y, 35, X_T, y_T, trainData)









"""DO HYPERPARAMS TUNING:"""
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.5, random_state=0)
    #
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                      'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # scores = ['precision', 'recall']
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    # clf = GridSearchCV(SVC(), tuned_parameters, scoring='%s_macro' % score)
    # clf.fit(X_train, y_train)
    # print("Best parameters set found on development set:")
    # print()
    # print(clf.best_params_)
    # print()
    # print("Grid scores on development set:")
    # print()
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()
    # print("Detailed classification report:")
    # print()
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # print()
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))
    # print()
"""DO HYPERPARAMS TUNING:"""


"""DO SCALING TO DATA:"""
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    #
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    #
    # X_T = scaler.transform(X_T)
"""DO SCALING TO DATA:"""


"""DO SHAP FOR DATA:"""
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
"""DO SHAP FOR DATA:"""




# function that iterates over params and check which is the best
def calcBestNumOfFolds(clf, X, y, n, test_data, test_label, trainData):
    # var to save max for 2015_2016
    max_acc = 0
    """--------------------------------- Feature Scaling ------------------------------------"""
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    # test_data = scaler.transform(test_data)

    X_train_max, X_test_max, y_train_max, y_test_max = 0,0,0,0

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
