#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
import collections
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

def getAccuracy(pred, labels_test):
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return acc


def classifySVM(features_train, labels_train, kernel_type, c):
    # train it on a smaller training dataset (this values are used to optimized the model)
    # features_train = features_train[:len(features_train) / 100]
    # labels_train = labels_train[:len(labels_train) / 100]
    # import the sklearn module for SVC
    from sklearn.svm import SVC
    # create classifier
    clf = SVC(kernel=kernel_type, C=c)
    # fit the classifier on the training features and labels
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time() - t0, 3), "s"
    # return the fit classifier
    return clf


def main():
    # kernel_type = "linear"
    # clf = classifySVM(features_train, labels_train, kernel_type)
    # c =  10.0 (0.616040955631), 100., 1000., and 10000
    c = 10000
    kernel_type = "rbf"
    clf = classifySVM(features_train, labels_train, kernel_type, c)
    t1 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time() - t1, 3), "s"

    answer10 = pred[10]
    answer26 = pred[26]
    answer50 = pred[50]

    print "Email 10:", answer10
    print "Email 26:", answer26
    print "Email 50:", answer50

    print "Totals:", collections.Counter(pred)

    acc = getAccuracy(pred, labels_test)
    print "Accuracy", acc


if __name__ == '__main__':
    main()