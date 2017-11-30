# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class IteratedBagging(object):
	def __init__(self,base_estimator=None,estimator_num=5,x_train=np.array([]), y_train=np.array([]),
                 x_test=np.array([]), y_test=np.array([])):
		self.base_estimator = base_estimator
		self.estimator_num = estimator_num
		self.x_train = x_train
		self.y_train = y_train
		# x_test 和 y_test 是输入的测试集
		self.x_test = x_test
		self.y_test = y_test


		self.estimators = []


	def check(self,Loss,loss_2):
		for i,loss in enumerate(Loss):
			if loss * 1.1 < loss_2 :
				return True
		return False

	# bootstrap sample
	def resample(self,length):
		idx = np.random.randint(0, length, size=(length))
		idx = idx[:int(len(idx) * 0.8)]
		return idx

	def train(self):
		length = len(self.x_train)
		L = []
		loss = []
		pre = 0
		y_true = 0
		for i in range(int(self.estimator_num)):
			reg_1 = self.base_estimator()
			sample_index = self.resample(length)
			reg_1.fit(self.x_train[sample_index],self.y_train[sample_index])
			L.append(reg_1)

		self.estimators.append(L)

		pre = self.predict(self.x_train)
		residual = self.y_train - pre
		loss_1 = np.sum(np.square(self.y_train - pre)) / len(self.x_train)
		loss.append(loss_1)
		while True:
			#pre = 0
			y_temp  = residual
			L = []
			for i in range(int(self.estimator_num)):
				reg_1 = self.base_estimator()
				sample_index = self.resample(length)
				reg_1.fit(self.x_train[sample_index],y_temp[sample_index])
				#pre += reg_1.predict(self.x_train)
				L.append(reg_1)

			self.estimators.append(L)
			#pre = pre/self.estimator_num
			pre = self.predict(self.x_train)
			residual = y_temp - pre
			loss_2 = np.sum(np.square(y_temp - pre)) / len(self.x_train)
			
			if self.check(loss,loss_2):
				break;
			loss.append(loss_2)
			#loss_1 = loss_2

	def predict(self,x_test):
		sum_row = np.zeros((x_test.shape[0],))
		sum_total = np.zeros((x_test.shape[0],))
		for i, stage in enumerate(self.estimators):
			sum_row = np.zeros((x_test.shape[0],))
			for j, estimator in enumerate(stage):
				s1 = estimator.predict(x_test)
				sum_row += s1
			sum_total += sum_row/self.estimator_num
		return sum_total
	def test (self):
		pass



