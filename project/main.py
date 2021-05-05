import dataInit
import KNearestNeighbors
import SupportVectorMachine
import dataAnalyze
# import GaussianNaiveBayes

from project import AdaBoost, KNearestNeighbors,dataInit

if __name__ == '__main__':
    trainData, testData = dataInit.init()
    # dataAnalyze.check_data(trainData)
    # AdaBoost Classifier Model:
    AdaBoost.runAdaBoost(trainData, testData)


    # GaussianNaiveBayes.naive_bayes_function(trainData, testData)

    # K - Nearest Neighbors Model:
    KNearestNeighbors.model_KNN(trainData, testData)
    # KNearestNeighbors.trainWithGridSearchCV(trainData, testData)

    # SVM
    # SupportVectorMachine.model_SVM(trainData, testData)

