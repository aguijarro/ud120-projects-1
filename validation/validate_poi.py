#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
from time import time
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

### it's all yours from here forward!


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
    clf = tree.DecisionTreeClassifier()
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
    #print "predicting time:", round(time() - t1, 3), "s"
    acc = getAccuracy(pred, labels_test)
    print "Accuracy", acc

if __name__ == '__main__':
    main()