#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 23:24:28 2019

@author: rahul
"""

# importing the libraries
import pandas as pd
from sklearn.metrics import  accuracy_score


dataset = pd.read_csv("heart.csv")



# extracting features
# “iloc” in pandas is used to select rows and columns by number

X = dataset.iloc[:, :-1]

y = dataset.iloc[:, -1]

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
# Scale X object
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
lgr_classifier = LogisticRegression(random_state = 0)
lgr_classifier.fit(X_train, y_train)

# Predicting test set result
y_pred_lgr = lgr_classifier.predict(X_test)

# Making confusion matrix to check number of correct predictions and incorrect predictions
from sklearn.metrics import confusion_matrix
lgr_cm = confusion_matrix(y_test, y_pred_lgr)

# finding percentage accuracy

print(accuracy_score(y_test, y_pred_lgr))

########################################################
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = knn_classifier.predict(X_test)

# Making the Confusion Matrix for KNN
knn_cm = confusion_matrix(y_test, y_pred_knn)

# finding percentage accuracy
print(accuracy_score(y_test, y_pred_knn))

###########################################################

# Fitting SVM to the Training set
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'linear', random_state = 0)
svm_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_svm = svm_classifier.predict(X_test)

# Making the Confusion Matrix for SVM
svm_cm = confusion_matrix(y_test, y_pred_svm)

# finding percentage accuracy
print(accuracy_score(y_test, y_pred_svm))
#######################################################

# Fitting Kernel SVM to the Training set
kernel_svm_classifier = SVC(kernel = 'rbf', random_state = 0)
kernel_svm_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_kernel_svm = kernel_svm_classifier.predict(X_test)

# Making the Confusion Matrix for kernel svm
kernel_svm_cm = confusion_matrix(y_test, y_pred_kernel_svm)

# finding percentage accuracy
print(accuracy_score(y_test, y_pred_kernel_svm))

#################################################################
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_naive_bayes = naive_bayes_classifier.predict(X_test)

# Making the Confusion Matrix for naive bayes
naive_bayes_cm = confusion_matrix(y_test, y_pred_naive_bayes)

# finding percentage accuracy
print(accuracy_score(y_test, y_pred_naive_bayes))

##################################################################

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
decision_tree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
decision_tree_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_decision_tree = decision_tree_classifier.predict(X_test)

# Making the Confusion Matrix for decision_tree
decision_tree_cm = confusion_matrix(y_test, y_pred_decision_tree)


# finding percentage accuracy
print(accuracy_score(y_test, y_pred_decision_tree))

######################################################################
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
random_forest_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_random_forest = random_forest_classifier.predict(X_test)

# Making the Confusion Matrix for random forest
random_forest_cm = confusion_matrix(y_test, y_pred_random_forest)

# finding percentage accuracy
print(accuracy_score(y_test, y_pred_random_forest))


