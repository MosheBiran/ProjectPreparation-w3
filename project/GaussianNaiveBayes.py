from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


def split_DataFrame_2_TrainAndTest(data):
    X_ = data.drop(["Result"], axis=1)
    y_ = data["Result"]
    return X_, y_


def print_ModelResult(y, model_prediction, test_name=''):
    print("---------------------------------" + str(test_name) + "---------------------------------------")
    print("Training data accuracy is", str(accuracy_score(y, model_prediction)), "%")
    print(confusion_matrix(y, model_prediction))
    print(classification_report(y, model_prediction))


def naive_bayes_function(train_data, test_data):

    model = GaussianNB()
    X_train_model, y_train_model = split_DataFrame_2_TrainAndTest(train_data)
    X_test_model, y_test_model = split_DataFrame_2_TrainAndTest(test_data)

    model.fit(X_train_model, y_train_model)

    training_predictions = model.predict(X_train_model)
    future_predictions = model.predict(X_test_model)

    print_ModelResult(y_train_model, training_predictions, "train")
    print_ModelResult(y_test_model, future_predictions, "test")
