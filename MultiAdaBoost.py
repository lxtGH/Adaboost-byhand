# -*- coding: utf-8 -*-
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import loadData


class MultiAdaBoost(object):
    def __init__(self, base_estimator=None, n_iteration=100, target=0.001, x_train=np.array([]), y_train=np.array([]),
                 x_test=np.array([]), y_test=np.array([])):
        self.base_estimator = base_estimator  ### 基分类器的类型
        self.n_iteration = n_iteration  ###	迭代次数等于基分类器的个数
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
        self.weights = [1] * len(self.x_train)
        self.bootstrap = range(0, len(self.x_train))

        self.iterations = []
        self.current_iteration = 0
        self._set_iterations()

    @staticmethod
    def _possion_sample(length):
        bootstrap = []
        for i in range(length):
            tmp = length + 1
            while tmp >= length:
                tmp = np.random.poisson(i, 1)
            bootstrap.append(tmp[0])
        return bootstrap

    # 设置停止 iteration
    def _set_iterations(self):
        n = int(float(self.n_iteration) ** 0.5)
        for i in range(n):
            self.iterations.append(int(((i + 1) * self.n_iteration + n - 1) / n))
        for i in range(self.n_iteration):
            self.iterations.append(self.n_iteration)

    def train_one_iteration(self, iteration):
        # 体现bagging的一步。如果当前 iteration 等于停止 iteration，说明当前 bagging 应该结束了，就重新采样，同时设置权重
        if self.iterations[self.current_iteration] == iteration:
            self.bootstrap = self._possion_sample(len(self.x_train))
            self.weights = [1] * len(self.x_train)
            self.current_iteration += 1
        clf = self.base_estimator()
        clf.fit(self.x_train[self.bootstrap], self.y_train[self.bootstrap])
        y_train_result = clf.predict(self.x_train[self.bootstrap])
        errors = (self.y_train[self.bootstrap] != y_train_result)
        error = np.sum(self.weights * errors) / len(self.x_train)
        # 如果误差太大，重新 sample
        if error > 0.5:
            self.bootstrap = self._possion_sample(len(self.x_train))
            self.weights = [1] * len(self.x_train)
            self.current_iteration += 1
            return
        # 如果误差太小，说明对当前 bootstrap 的数据集，我们表现的足够好，重新 sample
        elif error < 1e-5:
            self.beta.append(1e-10)
            self.bootstrap = self._possion_sample(len(self.x_train))
            self.weights = [1] * len(self.x_train)
            self.current_iteration += 1
        # 否则，常规 adaboost
        else:
            self.beta.append(np.log((1 - error) / error))
            self.weights = [0.5 * weight / error if errors[index] else 0.5 * weight / (1 - error)
                            for index, weight in enumerate(self.weights)]
            self.weights = [1e-8 if weight < 1e-8 else weight for weight in self.weights]

        self.estimators.append(clf)

    # train the data set
    def train(self):
        for i in range(self.n_iteration):
            self.train_one_iteration(i)

    # test operation
    def test(self):
        result = []
        for i in range(len(self.x_test)):
            result.append([])
        # 统计不同分类器针对的分类结果
        for index, estimator in enumerate(self.estimators):
            y_test_result = estimator.predict(self.x_test)
            for index2, res in enumerate(result):
                res.append([y_test_result[index2], np.log(1 / self.beta[index])])
        #
        final_result = []
        # 投票得出结果
        for res in result:
            dic = {}
            for r in res:
                if r[0] not in dic:  ## 记录每一个分类器对当前test实例的结果
                    dic[r[0]] = r[1]
                else:
                    dic[r[0]] = dic.get(r[0]) + r[1]

            final_result.append(sorted(dic, key=lambda x: dic[x])[-1])

        print(float(np.sum(final_result == self.y_test)) / len(self.y_test))

        return final_result
