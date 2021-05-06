from project import AdaBoost, KNearestNeighbors, dataInit, SupportVectorMachine, GaussianNaiveBayes, dataAnalyze

if __name__ == '__main__':
    trainData, testData = dataInit.init()

    # print(trainData.columns)
    # dataAnalyze.check_data(trainData)

    print("train Data Size:  " + str(len(trainData)))
    print("test Data Size:  " + str(len(testData)))

    # SVM
    print("\n**************************************************************")
    print("***********************   Model : SVM   **********************")
    print("**************************************************************")
    SupportVectorMachine.model_SVM(trainData, testData)


    # AdaBoost Classifier Model:
    print("\n**************************************************************")
    print("***************   Model : AdaBoost Classifier   **************")
    print("**************************************************************")
    AdaBoost.runAdaBoost(trainData, testData)


    # K - Nearest Neighbors Model:
    print("\n**************************************************************")
    print("*************   Model : K - Nearest Neighbors   **************")
    print("**************************************************************")
    KNearestNeighbors.model_KNN(trainData, testData)
    # KNearestNeighbors.trainWithGridSearchCV(trainData, testData)


    # Gaussian Naive Bayes Model:
    print("\n**************************************************************")
    print("**************   Model : Gaussian Naive Bayes   **************")
    print("**************************************************************")
    GaussianNaiveBayes.naive_bayes_function(trainData, testData)


