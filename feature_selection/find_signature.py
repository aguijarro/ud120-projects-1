#!/usr/bin/python

import pickle
import numpy
from time import time
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

#exercises 28
#vocab_list = vectorizer.get_feature_names()
#print len(vocab_list)
#print "Feature words 19671:", vocab_list[19671]
#print "Feature words 24321:", vocab_list[24321]
#print "Feature words 33201:", vocab_list[33201]
#print "Feature words 33614:", vocab_list[33614]

# exercise 29
# vocab_list = vectorizer.get_feature_names()
# print "Feature words 14343:", vocab_list[14343]


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
#features_train = features_train[:150].toarray()
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here


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



clf = classifyDTree(features_train, labels_train)
t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time() - t1, 3), "s"
acc = getAccuracy(pred, labels_test)
print "Accuracy", acc
print "Features number:", getNumberFeatures(features_train)
print "features_train: ", len(features_train)

# exercise 27 - 29
feature_importance = clf.feature_importances_
# get the most importance feature
x = 0
for i in feature_importance:
    if i != 0:
        print "importance", round(i,10)
        print "indice: ", x
    x = x + 1
