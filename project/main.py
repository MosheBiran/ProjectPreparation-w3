# import GaussianNaiveBayes
import time
import itertools
from project import AdaBoost, KNearestNeighbors, dataInit, SupportVectorMachine, dataAnalyze


def my_func(params):
    print(params)
if __name__ == '__main__':
    trainData, testData = dataInit.init()
    # AdaBoost Classifier Model:
    print("AdaBoost Classifier Model")
    AdaBoost.runAdaBoost(trainData, testData)

    # SVM
    print("SVM")
    SupportVectorMachine.model_SVM(trainData, testData)

    # loop over to find best combination of fields
    # counter_times=0
    # dict={}
    # list_of_fields = [True] * 20
    # list_of_fields[17] = False
    # list_of_fields[12] = False
    # for p in itertools.product([True, False], repeat=21):
    #     print(counter_times)
    #     counter_times += 1
    #     if p not in dict:
    #         for i in range(0,len(list_of_fields)):
    #             list_of_fields[i] = p[i]
    #             print(list_of_fields[i])
    #         print("***************")
    #         dict[p] = counter_times
    #         trainData, testData = dataInit.init(list_of_fields)
    #         SupportVectorMachine.model_SVM(trainData, testData)


        # GaussianNaiveBayes.naive_bayes_function(trainData, testData)

    # K - Nearest Neighbors Model:
    # KNearestNeighbors.model_KNN(trainData, testData)
    # KNearestNeighbors.trainWithGridSearchCV(trainData, testData)




# for the doch
# loop over to find best combination of fields
#     counter_times=0
#     dict={}
#     list_of_fields = [True] * 20
#     list_of_fields[17] = False
#     list_of_fields[12] = False
#     for p in itertools.product([True, False], repeat=21):
#         print(counter_times)
#         counter_times += 1
#         if p not in dict:
#             for i in range(0,len(list_of_fields)):
#                 list_of_fields[i] = p[i]
#                 print(list_of_fields[i])
#             print("***************")
#             dict[p] = counter_times
#             trainData, testData = dataInit.init(list_of_fields)
#             SupportVectorMachine.model_SVM(trainData, testData)
