import dataInit

import GaussianNaiveBayes
if __name__ == '__main__':
    trainData, testData = dataInit.init()
    # run AdaBoost
    AdaBoost.runAdaBoost(trainData,testData)


    # GaussianNaiveBayes.naive_bayes_function(trainData, testData)

    # K - Nearest Neighbors Model:
    KNearestNeighbors.model_KNN(trainData, testData)

