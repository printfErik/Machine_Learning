# -*- coding: utf-8 -*-
"""hw2_logistic.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-6isUgjuVeRvpyrAjcY0oecvAqUyAH_9
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA


def get_next_train_valid(X_shuffled, y_shuffled, itr):
  size = X_shuffled.shape[0]

  # select the correct slice of valid data set with each fold has equal length 29
  X_valid = X_shuffled[itr * (size//5) : (itr+1)*(size//5),:]
  y_valid = y_shuffled[itr * (size//5) : (itr+1)*(size//5)]

  # delete valid set to get training set
  X_train = np.delete(X_shuffled, np.s_[itr * (size//5) : (itr+1) * (size//5) : 1], 0)
  y_train = np.delete(y_shuffled, np.s_[itr * (size//5) : (itr+1) * (size//5) : 1], 0)
  return X_train, y_train, X_valid, y_valid


def train(X_train,y_train):
  # add 1s to the tail of X_train since we need to deal with the model intercept
  size = X_train.shape[0]
  one = np.ones((size,1))
  X_new = np.concatenate((X_train,one),axis = 1)

  # choose learning_rate
  learning_rate = 1

  # initialize the weights randomly to begin with
  w = np.random.rand(3)

  # gradient descent 
  for i in range(500):
    pre_w = w
    for j in range(size):
      wT = w.transpose()
      temp1 = - np.matmul(wT,X_new[j])
      sigma = 1/(1+np.exp(temp1))
      gradient = (y_train[j] - sigma) * X_new[j]
      w = w + learning_rate * gradient
    
    # test small amount of changes between last w and current w
    if LA.norm(w-pre_w,ord = 2)<0.01:
      #print("break!!!!!!!")
      return w[:2], w[2] 

  # seperate weights and intercept
  model_weights = w[:2]
  model_intercept = w[2]
  return model_weights, model_intercept


def predict(X_valid,model_weights,model_intercept):
  size = X_valid.shape[0]
  y_predict_class = np.zeros(size)

  # if the regression larger than 0.5, then y goes to 1; otherwise y goes to 0
  for i in range(size):
    if (1/(1+np.exp(- np.matmul(X_valid[i],model_weights)-model_intercept))) >= 0.5:
      y_predict_class[i] = 1
    else:
      y_predict_class[i] = 0
  
  return y_predict_class


if __name__ == "__main__":

  # read data
  features_X = pd.read_csv('IRISFeat.csv').values
  target_y = pd.read_csv('IRISlabel.csv').values

  # shuffle data
  to_be_shuffled = np.concatenate((features_X,target_y),axis = 1)

  np.random.shuffle(to_be_shuffled)

  X_shuffled = to_be_shuffled[:,:2]
  y_shuffled = to_be_shuffled[:,2]

  # create vectors for recoding error rate
  error_rate_train = np.ones(5)
  error_rate_valid = np.ones(5)
  error_rate = np.ones(5)

  # create vectors for recoding number of errors
  n_error_valid = np.ones(5)
  n_error_train = np.ones(5)
  n_error_total = np.ones(5)

  # create vectors for recoding confusion matrix information
  True_Positives = 0
  False_Positives = 0
  False_Negatives = 0
  True_Negatives = 0

  # 5 folds cross validation
  for iteration in range(5):
    X_train, y_train, X_valid, y_valid = get_next_train_valid(X_shuffled, y_shuffled, iteration)

    model_weights, model_intercept = train(X_train,y_train)

    # predict validation data set
    predict_valid = predict(X_valid,model_weights,model_intercept)
    n_error_valid[iteration] = np.sum(np.absolute(predict_valid - y_valid))
    error_rate_valid[iteration] = n_error_valid[iteration] / y_valid.shape[0] 

    # recording confusion matrix for validation error data
    for i in range(y_valid.shape[0]):
      if predict_valid[i] == 1 and y_valid[i] == 1:
        True_Positives +=1
      elif predict_valid[i] == 1 and y_valid[i] == 0:
        False_Positives +=1
      elif predict_valid[i] == 0 and y_valid[i] == 1:
        False_Negatives +=1
      elif predict_valid[i] == 0 and y_valid[i] == 0:
        True_Negatives +=1

    # predict training data set
    predict_train = predict(X_train,model_weights,model_intercept)
    n_error_train[iteration] = np.sum(np.absolute(predict_train - y_train))
    error_rate_train[iteration] = n_error_train[iteration] / y_train.shape[0] 

    #total error rate
    n_error_total[iteration] = n_error_train[iteration] + n_error_valid[iteration]
    error_rate[iteration] = n_error_total[iteration] / (y_valid.shape[0] + y_train.shape[0])


  # show result
  print("number of errors for valid set: ")
  print(n_error_valid)
  print("number of errors for train set: ")
  print(n_error_train)
  print("number of errors for all set: ")
  print(n_error_total)

  print("True_Positives =")
  print(True_Positives)
  print("False_Positives =")
  print(False_Positives)
  print("False_Negatives =")
  print(False_Negatives)
  print("True_Negatives =")
  print(True_Negatives)

  # bar chart plotting
  x = np.arange(5)  
  width = 0.35 
  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width/3, error_rate_train, width, label='Training')
  rects2 = ax.bar(x, error_rate_valid, width, label='Validation')
  rects3 = ax.bar(x + width/3, error_rate, width, label='Total')

  
  ax.set_ylabel('Error Rate')
  ax.set_xlabel('k Fold')
  ax.set_title('Error Rate by Folds and Data Set')
  ax.set_xticks(x)
  ax.set_xticklabels(('1','2','3','4','5'))
  ax.legend()

  fig.tight_layout()

  plt.show()