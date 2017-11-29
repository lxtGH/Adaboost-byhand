# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor

import numpy as np
boston = datasets.load_boston()
X = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn=KNeighborsRegressor()

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

#print (y_pred)


print(np.sum(np.square(y_pred - y_test))/len(y_test))



bgg = BaggingRegressor(base_estimator = KNeighborsRegressor())

bgg.fit(x_train,y_train)

y_pred = bgg.predict(x_test)

print(y_pred.shape)

print(np.sum(np.square(y_pred - y_test))/len(y_test))

print(y_pred)