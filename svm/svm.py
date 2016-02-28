from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = load_iris()
clf = svm.SVC(kernel='linear')
clf.fit(iris.data, iris.target) 


'''
we will use the training dataSet as test dataSet
'''
y_pred =clf.predict(iris.data)
accuracy =  accuracy_score(iris.target ,y_pred)
print("the accuracy of the of the naive bayes by using iris data dataSet  %f" % (accuracy))
