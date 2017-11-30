#from IteratedBaggingWithRandom import IteratedBagging
from IteratedBagging_correct import IteratedBagging
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor

import numpy as np
boston = datasets.load_boston()

X = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


bagging = IteratedBagging(DecisionTreeRegressor,100,x_train,y_train,x_test,y_test)

bagging.train()


y_pred = bagging.predict(x_test)


loss = (np.sum(np.square(y_pred - y_test))/len(y_test))
print("loss1: ",loss,"length:",len(bagging.estimators))


t = DecisionTreeRegressor()

t.fit(x_train,y_train)


y_pred = t.predict(x_test)


loss = (np.sum(np.square(y_pred - y_test))/len(y_test))
print("loss2: ",loss)


b = BaggingRegressor()

b.fit(x_train,y_train)

y_pred = b.predict(x_test)

loss = (np.sum(np.square(y_pred - y_test))/len(y_test))
print("loss2: ",loss)