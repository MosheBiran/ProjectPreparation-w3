import dataInit
import KNearestNeighbors
import SupportVectorMachine

import GaussianNaiveBayes
if __name__ == '__main__':
    trainData, testData = dataInit.init()


    # GaussianNaiveBayes.naive_bayes_function(trainData, testData)

    # K - Nearest Neighbors Model:

    # KNearestNeighbors.model_KNN(trainData, testData)
    KNearestNeighbors.trainWithGridSearchCV(trainData, testData)
    #KNearestNeighbors.model_KNN(trainData, testData)

    #SVM
    SupportVectorMachine.model_SVM(trainData, testData)

