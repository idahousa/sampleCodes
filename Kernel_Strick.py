#Date-of-code: 2018-03-22
#Author: datnt
#Descriptions:
#Example of using LinearSVC for linear support vector machine classification
#Using LinearSVC for multi-classes classification problem
#PolynomialFeatures(degree=1) => LinearSVC
#PolynomialFeatures(degree=n) with n>1 => Polynomial kernel.
#Change the degree value to see the effect of kernel in SVM method.
#Note: Data must be normalized before using (such as the use of StandardScaler())
#      Please use loss = 'hinge' for better loss function.
#=================================================================================================================#
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
#load data and train model
iris = datasets.load_iris()
data = iris.data
label = iris.target.astype(np.float64)
clf = Pipeline((('Poly',PolynomialFeatures(degree=2)),('scaler',StandardScaler()),('clf',LinearSVC(C=1,loss='hinge'))))
clf.fit(data,label)
#Testing
print('Distance to hyperplance is: {}'.format(clf.decision_function([[0.1, 0.2, 0.5, 1.7]])))
print('Class label is: {}'.format(clf.predict([[0.1, 0.2, 0.5, 1.7]])))
print('Training accuracy is: {}'.format(clf.score(data,label)))
print(clf)
#End of code
#=================================================================================================================#
