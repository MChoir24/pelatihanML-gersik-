#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to classify the terrain data.

    The objective of this exercise is to recreate the decision
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """


import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from ClassifyNB import classify
from sklearn.metrics import accuracy_score


features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them

clf = classify(features_train, labels_train)

pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print ("Accuracy: ", accuracy)

### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_train, labels_train).show()
