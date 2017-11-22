# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def loadData_your():
	pass

def loadData():
	data = []
	label = []
	dic = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
        ,'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'draw': 17}
	with open("./Data/krkopt.data") as f:
		for line in f:
			tokens = line.strip().split(',')
			L = []
			for tk in tokens[:-1]:
				if tk.isdigit():
					tmp = float(tk)
				else:
					tmp = float(ord(tk)-ord('a'))
				L.append(tmp)
			data.append(L)
			label.append(dic[tokens[-1]])
	X = np.array(data)
	y = np.array(label)
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

	return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = loadData()

print(x_train.shape,x_test.shape)


print(y_train.shape,y_test.shape)