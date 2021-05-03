import dataInit
import GaussianNaiveBayes
import preprocessing

if __name__ == '__main__':
    trainData, testData = dataInit.init()
    GaussianNaiveBayes.naive_bayes_function(trainData, testData)
    # preprocessing.check_data()
    # dataInit.temp()
