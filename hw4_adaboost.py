# -*- coding: utf-8 -*-
"""hw4_adaboost.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DtzNziqK7o05CoSDBCXomroPTUc8GOri
"""

from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# read data
train_data = pd.read_csv('cancer_train.csv').to_numpy()
test_data = pd.read_csv('cancer_test.csv').to_numpy()

train_feature = train_data[:,:-1]
train_label = train_data[:,-1]

test_feature = test_data[:,:-1]
test_label = test_data[:,-1]


# training...
def train(feature_set,label_set,T):
  
  sample_size = len(label_set)
  
  dt_list = []
  alpha_list = []
  sample_weight = np.ones(sample_size) / sample_size
  output = np.random.choice(sample_size, sample_size, p=sample_weight)
  feature_set = feature_set[output]
  label_set = label_set[output] 
  for i in range(T):
    dt = tree.DecisionTreeClassifier(criterion = 'gini',max_depth = 1)
    dt.fit(feature_set,label_set,sample_weight = sample_weight)
    pred = dt.predict(feature_set)
    
    dt_error = np.sum((pred != label_set) * sample_weight)
    if dt_error >0.5:
      sample_weight = - sample_weight
      pred = dt.predict(feature_set)
      dt_error = np.sum((pred != label_set) * sample_weight)

    alpha_t = 0.5 * np.log((1.-dt_error)/dt_error)


    sample_weight = sample_weight * np.exp(-alpha_t * pred * label_set)
    sample_weight = sample_weight / np.sum(sample_weight)

    dt_list.append(dt)
    alpha_list.append(alpha_t.copy())

  return dt_list,alpha_list

# predict
def predict(dt_list, alpha_list, feature_set,label_set,T):
  sum1 = np.zeros(feature_set.shape[0])
  for i in range(T):
    alpha = alpha_list[i]
    clf = dt_list[i]
    sum1+=(alpha * clf.predict(feature_set))
  fboost_train_pred = np.sign(sum1)

  error_rate = np.sum(fboost_train_pred!=label_set)/label_set.shape[0]
  accuracy = 1 - error_rate
  return error_rate
  
# For problem 5 (b) 
train_error_rates = []
test_error_rates = []
for t in range(1,1000):
  dt_list,alpha_list = train(train_feature,train_label,t)
  train_error = predict(dt_list,alpha_list,train_feature,train_label,t)
  test_error = predict(dt_list,alpha_list,test_feature,test_label,t)
  train_error_rates.append(train_error)
  test_error_rates.append(test_error)

plt.plot(train_error_rates,label='train')
plt.plot(test_error_rates,label='test')
plt.xlabel('Number of Weak Learners')
plt.ylabel('Error Rate')
plt.legend()
plt.show()

# For problem 5 (c)
def margin(X,y,dt_list,alpha_list,T):
  sum1 = np.zeros(X.shape[0])
  total_alpha = 0;
  for i in range(T):
    alpha = alpha_list[i]
    total_alpha+=alpha
    clf = dt_list[i]
    sum1+=(alpha * clf.predict(X))
  marginX = y * sum1 /  total_alpha
  return marginX

margin25 = margin(train_feature,train_label,dt_list,alpha_list,25)
margin50 = margin(train_feature,train_label,dt_list,alpha_list,25)
margin75 = margin(train_feature,train_label,dt_list,alpha_list,75)
margin100 = margin(train_feature,train_label,dt_list,alpha_list,100)

fig, axs = plt.subplots(2,2,tight_layout = True)
axs[0][0].hist(margin25)
axs[0][1].hist(margin50)
axs[1][0].hist(margin75)
axs[1][1].hist(margin100)


