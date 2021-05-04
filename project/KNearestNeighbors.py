import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html



def model_KNN(trainData, testData):

    """--------------------------------- Splitting The Train Data ------------------------------------"""


    X = trainData.iloc[:, :-1].values
    y = trainData.iloc[:, len(trainData.columns) - 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    """--------------------------------- Feature Scaling ------------------------------------"""

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    """--------------------------------- Train Of The Model ------------------------------------"""


    classifier = KNeighborsClassifier(n_neighbors=21)
    classifier.fit(X_train, y_train)


    """--------------------------------- Prediction And Evaluation------------------------------------"""

    y_predict = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_predict))
    print(classification_report(y_test, y_predict))
    print("\n**************************\n")
    print("Training data accuracy is", str(accuracy_score(y_test, y_predict)), "%")
    print("\n**************************\n")


    """--------------------------------- Splitting The Test Data ------------------------------------"""


    season_15_16_features = testData.iloc[:, :-1].values
    season_15_16_Result = testData.iloc[:, len(testData.columns) - 1].values


    """--------------------------------- Feature Scaling ------------------------------------"""

    scaler = StandardScaler()
    scaler.fit(season_15_16_features)

    season_15_16_features = scaler.transform(season_15_16_features)


    """--------------------------------- Prediction And Evaluation------------------------------------"""


    prediction_on_15_16 = classifier.predict(season_15_16_features)

    print("\n**************************\n")
    print("Seasons 2015/2016 data accuracy is", str(accuracy_score(season_15_16_Result, prediction_on_15_16)), "%")
    print("\n**************************\n")


    """--------------------------------- Check Err KNN ------------------------------------"""

    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')

    plt.show()

