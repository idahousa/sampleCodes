import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
data  = iris.data[:,2:]
label = (iris.target==2).astype(np.float64)
clf = Pipeline((('scaler',StandardScaler()),('clf',LinearSVC(C=1,loss='hinge'))))
clf.fit(data,label)
#Testing
print('Distance to hyperplance is: {}'.format(clf.decision_function([[0.5, 1.7]])))
print('Class label is: {}'.format(clf.predict([[0.5, 1.7]])))
print('Training accuracy is: {}'.format(clf.score(data,label)))
print(clf)
