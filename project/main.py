from project import AdaBoost, KNearestNeighbors, dataInit, SupportVectorMachine, GaussianNaiveBayes, dataAnalyze

if __name__ == '__main__':
    trainData, testData = dataInit.init()

    print(trainData.columns)
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
