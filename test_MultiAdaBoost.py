# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from utils import loadData
from MultiAdaBoost import MultiAdaBoost
from Adaboost import AdaBoostWithBoostrap
x_train, x_test, y_train, y_test = loadData()


print("test set for the adaboost with boostrap:")

adaboost = AdaBoostWithBoostrap(base_estimator=tree.DecisionTreeClassifier, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

adaboost.train()

adaboost.test()

print("test set for the multiAdaboost:")

multiAdaboost = MultiAdaBoost(base_estimator=tree.DecisionTreeClassifier, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

multiAdaboost.train()

multiAdaboost.test()







