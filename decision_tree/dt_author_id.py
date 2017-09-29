#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###


#########################################################

def getNumberFeatures(features_train):
    return len(features_train[0])

def getAccuracy(pred, labels_test):
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return acc


def classifyDTree(features_train, labels_train):
    # import the sklearn module for GaussianNB
    from sklearn import tree
    # create classifier
    clf = tree.DecisionTreeClassifier(min_samples_split=40)
    # fit the classifier on the training features and labels
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time() - t0, 3), "s"
    # return the fit classifier
    return clf


def main():
    clf = classifyDTree(features_train, labels_train)
    t1 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time() - t1, 3), "s"
    acc = getAccuracy(pred, labels_test)
    print "Accuracy", acc
    print "Features number:", getNumberFeatures(features_train)

if __name__ == '__main__':
    main()