import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics



def model_SVM(trainData, testData):
    X = trainData.iloc[:, :-1].values
    y = trainData.iloc[:, len(trainData.columns) - 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # making the instance
    model = svm.SVC()
    # learning
    model.fit(X_train, y_train)
    # Prediction
    prediction = model.predict(X_test)
    # evaluation(Accuracy)
    print("Accuracy:", metrics.accuracy_score(y_test, prediction))
    # evaluation(Confusion Metrix)
    print("Confusion Metrix:\n", metrics.confusion_matrix(prediction, y_test))



    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))
    print("\n**************************\n")
    print("Training data accuracy is", str(accuracy_score(y_test, prediction)), "%")
    print("\n**************************\n")

# from the matzeget of the hartzaa
