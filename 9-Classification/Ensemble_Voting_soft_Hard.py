# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:05:24 2025

@author: Ashlesha
"""

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')

from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data[:, :3], iris.target  # taking entire data as training data

clf1 = LogisticRegression()
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()


print("After five fold cross validation")
labels = ['Logistic Regression', 'Random Forest model', 'Naive Bayes model']

for clf, label in zip([clf1, clf2, clf3], labels):
    scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: ", scores.mean(), "for", label)

voting_clf_hard = VotingClassifier(estimators=[
    (labels[0], clf1),
    (labels[1], clf2),
    (labels[2], clf3)
], voting='hard')

voting_clf_soft = VotingClassifier(estimators=[
    (labels[0], clf1),
    (labels[1], clf2),
    (labels[2], clf3)
], voting='soft')

labels_new = ['Logistic Regression', 'Random Forest model', 'Naive Bayes model',
              'Hard Voting Classifier', 'Soft Voting Classifier']

# Names for all five models (3 base + 2 ensemble voting models)
for clf, label in zip([clf1, clf2, clf3, voting_clf_hard, voting_clf_soft], labels_new):
    scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: ", scores.mean(), "for", label)
