from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification


def runAdaBoost(trainData):
    t = trainData["HomeTeamResult"]
    t1 = trainData.iloc[:, : -1]
    X = t1[:5000]
    y = t[:5000]
    X_test = t1[5000:]
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)
    AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.predict(X_test)
    clf.score(X, y)

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
