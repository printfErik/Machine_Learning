# -*- coding: utf-8 -*-
"""hw2_svm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lS_tG3_ZxXZi8O7YAJJjUdSYjeU_ZfqD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt
cvxopt.solvers.options['show_progress'] = False

# training weights
def svmfit(X, y, c):
  size = np.shape(X)[0]
  R = cvxopt.matrix(np.matmul(X.T,np.diag(y)))
  # define P,q,G,h for corresponding parameter in qp solver
  P = cvxopt.matrix(np.matmul(R.T,R))
  q = -cvxopt.matrix(np.ones(size))
  G = cvxopt.matrix(np.concatenate((np.eye(size),-np.eye(size))))
  h = cvxopt.matrix(np.concatenate((np.ones(size)*c,np.zeros(size))))
  ans = cvxopt.solvers.qp(P,q,G,h)
  lamda = ans['x']
  weight = np.matmul(R,lamda)
  return weight


def predict(X,weight):
  label = np.sign(np.matmul(X,weight))
  return label


def k_fold_cv(train_data, test_data, k, c):
  X_shuffled = train_data[:,:2]
  y_shuffled = train_data[:,2]

  X_test = test_data[:,:2]
  y_test = test_data[:,2]

  X_train, y_train, X_valid, y_valid = get_next_train_valid(X_shuffled, y_shuffled, k)

  # train model
  weight = svmfit(X_train,y_train,c)

  # model accuracy for trainning set
  train_label = predict(X_train,weight)
  train_label = np.reshape(train_label,len(X_train))
  train_accuracy = np.sum(train_label == y_train)/len(X_train)
 
  # model accuracy for validation set
  cv_label = predict(X_valid,weight)
  cv_label = np.reshape(cv_label,len(X_valid))
  cv_accuracy = np.sum(cv_label == y_valid)/len(X_valid)

  # model accuracy for test set
  test_label = predict(X_test,weight)
  test_label = np.reshape(test_label,len(X_test))
  test_accuracy = np.sum(test_label == y_test)/len(X_test)


  return train_accuracy, cv_accuracy, test_accuracy

# from hw1
def get_next_train_valid(X_shuffled, y_shuffled, itr):
  size = X_shuffled.shape[0]

  # select the correct slice of valid data set with each fold has equal length 29
  X_valid = X_shuffled[itr * (size//10) : (itr+1)*(size//10),:]
  y_valid = y_shuffled[itr * (size//10) : (itr+1)*(size//10)]

  # delete valid set to get training set
  X_train = np.delete(X_shuffled, np.s_[itr * (size//10) : (itr+1) * (size//10) : 1], 0)
  y_train = np.delete(y_shuffled, np.s_[itr * (size//10) : (itr+1) * (size//10) : 1], 0)
  return X_train, y_train, X_valid, y_valid



samples = pd.read_csv("hw2data.csv",header = None)
#print(np.shape(samples))
data = np.array(samples)
np.random.shuffle(data)
#print(np.shape(data))

# seperate test and cross-validation data
test_data = data[:np.shape(data)[0]//5]
cross_data = data[np.shape(data)[0]//5:]


C =  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
logC = [-4,-3,-2,-1,0,1,2,3]
train_acc = np.zeros(8)
cv_acc = np.zeros(8)
test_acc = np.zeros(8)


# compute average accuracy
for i in range(8):
  for j in range(10):
    train_accuracy, cv_accuracy, test_accuracy = k_fold_cv(cross_data,test_data,j,C[i])
    train_acc[i] += train_accuracy
    cv_acc[i] += cv_accuracy
    test_acc[i] += test_accuracy
  train_acc[i] = train_acc[i]/10
  cv_acc[i] = cv_acc[i]/10
  test_acc[i] = test_acc[i]/10


# plot line chart
plt.title("Accuracy Rate by Different C Value in Log Scale")
plt.xlabel('log(C)')
plt.ylabel('Accuracy Rate')
plt.plot(logC,  cv_acc, 'r-',label = 'Average accuracy rate of validation')   
plt.plot(logC,  train_acc, 'b-',label = 'Average accuracy rate of training')  
plt.plot(logC, test_acc, 'g-', label = 'Accuracy rate of testing')
plt.legend()
plt.show()

