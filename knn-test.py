import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
print('Number of classes:', len(np.unique(iris_Y)))
print('Number of data points:', len(iris_Y))

#X0 = iris_X[iris_Y == 0,:]
#print ('\nSamples from class 0:\n', X0[:5,:])

#X1 = iris_X[iris_Y == 1,:]
#print ('\nSamples from class 1:\n', X1[:5,:])

#X2 = iris_X[iris_Y == 2,:]
#print ('\nSamples from class 2:\n', X2[:5,:])

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=50)

print('Trainging size: ', len(y_train))
print('Test size: ', len(y_test))

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Print results for 20 test data points: ')
print('Predicted labels: ', y_pred[:20])
print('Ground truth:     ', y_test[:20])

print('Accuracy of 10NN: ', 100*accuracy_score(y_test, y_pred))
