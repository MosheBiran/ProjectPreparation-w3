from project import dataInit, AdaBoost

if __name__ == '__main__':
    trainData, testData = dataInit.init()
    # run AdaBoost
    AdaBoost.runAdaBoost(trainData)
