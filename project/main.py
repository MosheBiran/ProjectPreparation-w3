import dataInit
import GaussianNaiveBayes
if __name__ == '__main__':
    trainData, testData = dataInit.init()
    GaussianNaiveBayes.naive_bayes_function(trainData, testData)

    # dataInit.temp()