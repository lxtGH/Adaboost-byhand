from IteratedBaggingWithRandom import IteratedBagging

from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
boston = datasets.load_boston()

X = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


bagging = IteratedBagging(DecisionTreeRegressor,5,x_train,y_train,x_test,y_test)

bagging.train()

print(len(bagging.estimators))

y_pred = bagging.predict(x_test)

print(y_pred.shape)

print(y_test.shape)

loss = (np.sum(np.square(y_pred - y_test))/len(y_test))
print("loss: ",loss)