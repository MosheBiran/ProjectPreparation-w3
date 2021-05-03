import dataInit
import KNearestNeighbors

if __name__ == '__main__':
    trainData, testData = dataInit.init()

    # K - Nearest Neighbors Model:
    KNearestNeighbors.model_KNN(trainData, testData)

