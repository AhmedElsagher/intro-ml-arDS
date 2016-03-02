from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score


iris = load_iris()
gnb = GaussianNB()
gnb.fit(iris.data, iris.target)


'''
we will use the training dataSet as test dataSet
'''
y_pred =gnb.predict(iris.data)
accuracy =  accuracy_score(iris.target ,y_pred)
print("the accuracy of the of the naive bayes by using iris data dataSet  %f" % (accuracy))