#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/taekwon8290/OSS_project2_scikit-learn/tree/main

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, RandomForestClassifier
from sklearn.svm import SVC

def load_dataset(dataset_path):
  data_df = pd.read_csv(dataset_path)
  return data_df
	#To-Do: Implement this function

def dataset_stat(dataset_df):	
  dataset_df.shape[1]
  n_feats = len(dataset_df.colums)-1
  #target열을 제외한 나머지 열의 수(feature수)
  target = list(dataset_df['target'].value_counts())
  n_class0 = target[0]
  n_class1 = target[1]
  #target열에서 0과 1을 분류
  return n_feats, n_class0, n_class1
	#To-Do: Implement this function

def split_dataset(dataset_df, testset_size):
  x = dataset_df.drop(columns = "target", axis = 1)
  y = dataset_df["target"]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testset_size)
  return x_train, x_test, y_train, y_test
	#To-Do: Implement this function

def decision_tree_train_test(x_train, x_test, y_train, y_test):
  dt_cls = DecisionTreeClassifier()
  dt_cls.fit(x_train, y_train)

  accuracy = accuracy_score(y_test, dt_cls.predict(x_test))
  precision = precision_score(y_test, dt_cls.predict(x_test)) 
  recall = recall_score(y_test, dt_cls.predict(x_test))
  return accuracy, precision, recall
	#To-Do: Implement this function

def random_forest_train_test(x_train, x_test, y_train, y_test):
  rf_cls = RandomForestClassifier()
  rf_cls.fit(x_train, y_train)

  accuracy = accuracy_score(y_test, rf_cls.predict(x_test))
  precision = precision_score(y_test, rf_cls.predict(x_test)) 
  recall = recall_score(y_test, rf_cls.predict(x_test))
  return accuracy, precision, recall
	#To-Do: Implement this function

def svm_train_test(x_train, x_test, y_train, y_test):
  svm_cls = SVC()
  svm_cls.fit(x_train, y_train)
  
  accuracy = accuracy_score(y_test, svm_cls.predict(x_test))
  precision = precision_score(y_test, svm_cls.predict(x_test)) 
  recall = recall_score(y_test, svm_cls.predict(x_test))
  return accuracy, precision, recall
	#To-Do: Implement this function

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
