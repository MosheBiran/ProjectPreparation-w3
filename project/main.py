# import GaussianNaiveBayes

from project import AdaBoost, KNearestNeighbors,dataInit

if __name__ == '__main__':
    trainData, testData = dataInit.init()
    # AdaBoost Classifier Model:
    AdaBoost.runAdaBoost(trainData,testData)


    # GaussianNaiveBayes.naive_bayes_function(trainData, testData)

    # K - Nearest Neighbors Model:
    # KNearestNeighbors.model_KNN(trainData, testData)

