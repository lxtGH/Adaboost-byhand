# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MultiAdaBoost(object):
	def __init__(self, base_estimator=None, n_iteration=100, target=0.001, x_train=np.array([]), y_train=np.array([]), x_test=np.array([]), y_test=np.array([])):
		self.base_estimator = base_estimator ### 基分类器的类型
		self.n_iteration= n_iteration ###	迭代次数等于基分类器的个数
		self.target = target
		# adaboost 弱分类器的权重
		self.beta = []
		# adaboost 的多个弱分类器
		self.estimators = []
		# x_train 和 y_train 是输入的训练集
		self.x_train = x_train
		self.y_train = y_train
		# x_test 和 y_test 是输入的测试集
		self.x_test = x_test
		self.y_test = y_test
		# adaboost 的权重
		self.weights = [1]*len(self.x_train)
		self.bootstrap = range(0, len(self.x_train))


	
