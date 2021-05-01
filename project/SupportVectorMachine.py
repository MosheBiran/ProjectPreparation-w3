import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
X, y = np.arange(10).reshape((30, 2)), range(30)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42)
clf = svm.SVC(kernel=‘rbf)
clf.fit(X_train, y_train)
clf.predict(X_test)

#from the matzeget of the hartzaa