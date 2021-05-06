import numpy as np
import shap
from matplotlib import pyplot as plt
from matplotlib.pyplot import show
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


def model_SVM(trainData, testData):
    X = trainData.iloc[:, :-1].values
    y = trainData.iloc[:, len(trainData.columns) - 1].values

    X_T = testData.iloc[:, :-1].values
    y_T = testData.iloc[:, len(testData.columns) - 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_T = scaler.transform(X_T)



    # making the instance
    model = svm.LinearSVC()
    # learning
    model.fit(X_train, y_train)
    # Prediction
    prediction = model.predict(X_test)
    # evaluation(Accuracy)
    print("Accuracy:", metrics.accuracy_score(prediction, y_test))
    # evaluation(Confusion Metrix)
    print("Confusion Metrix:\n", metrics.confusion_matrix(prediction, y_test))
    prediction_test = model.predict(X_T)
    print("Accuracy 2015_2016:", metrics.accuracy_score(prediction_test, y_T))



    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("\n**************************\n")
    print("Training data accuracy is", str(accuracy_score(y_test, prediction)), "%")
    print("\n**************************\n")

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


