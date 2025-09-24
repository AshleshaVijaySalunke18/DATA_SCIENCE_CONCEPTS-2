# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:07:34 2025

@author: Ashlesha
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# ########## Loading the dataset
email_data = pd.read_csv("c:/Data-Science/9-Classification/sms_raw_NB.csv", encoding="ISO-8859-1")

# These SMS are in text form, open the data frame and view it
# ### Cleaning the data
# The function tokenizes the text and removes words with fewer than 4 characters.
import re

def cleaning_text(i):
    i = re.sub("[^A-Za-z""]+", " ", i).lower()
    w = []
    # Every thing else A to Z and a to z is going to convert into space
    # We will take each row and tokenize
    for word in i.split(" "):
        if len(word) > 3:
            w.append(word)
    return " ".join(w)

# Testing above function with sample text
cleaning_text("Hope you are having good week. just checking")
cleaning_text("hope i can i understand your feeling12")
cleaning_text("Hi how are you")

# Note: The data frame size is 5559 × 2, now removing empty rows
#removing  empty rows
email_data = email_data.loc[email_data.text != " ", :]
email_data.shape

# You can use count vectorizer which directly converts a collection of text documents to a matrix of token counts
# First we will split the data
from sklearn.model_selection import train_test_split
email_train, email_test = train_test_split(email_data, test_size=0.2)

# Splits each email into a list of words.
## Creating matrix of token count for entire dataframe
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email text into word count matrix format
# CountVectorizer: Converts the emails into a matrix of token counts.
# .fit(): Learns the vocabulary from the text data.
# .transform(): Converts text data into a token count matrix.

emails_bow = CountVectorizer(analyzer=split_into_words).fit(email_data.text)

# Defining BOW for all data frames
all_emails_matrix = emails_bow.transform(email_data.text)
train_email_matrix = emails_bow.transform(email_train.text)

# For testing messages
test_email_matrix = emails_bow.transform(email_test.text)

# Learning Term weighting and normalizing entire emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)

#preparing TFIDF for  mails
train_tfidf=tfidf_transformer.transform(train_email_matrix)
train_email_matrix

test_tfidf=tfidf_transformer.transform(test_email_matrix)
test_tfidf.shape

#### Now apply to naive bayes
from sklearn.naive_bayes import MultinomialNB as MB

classifier_mb = MB()
classifier_mb.fit(train_tfidf, email_train.type)
# email_train.type: This is the column in the training dataset
# (email_train) that contains the target labels,
# which specify whether each message is spam or ham (non-spam).
# The .type attribute refers to that specific column in the email_train data
# training data prepared in terms of tfidf and labels of corresponding training data

### evaluation on test data
test_pred_m = classifier_mb.predict(test_tfidf)

## calculating accuracy
accuracy_test_m = np.mean(test_pred_m == email_test.type)
accuracy_test_m

# Evaluation on Test Data accuracy matrix
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m,email_test.type)

######evaluation on test data accurracy matrix

#training Data Accuracy

train_pred_m=classifier_mb.predict(train_tfidf)
accuracy_train_m=np.mean(train_pred_m==email_train.type)
accuracy_train_m

#Test data (with laplace smoothing):This acuracy is
#computed after appling laplace smoothing (with alpha=3)
#to the navie bayed model
# Interpretation: Smoothing helps avoid issues when encountering
# words in the test data that were not seen in the training data
# (zero-frequency problem)

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, email_train.type)
## accuracy after tuning

test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == email_test.type)
accuracy_test_lap
accuracy_score(test_pred_lap, email_test.type)

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, email_test.type)
pd.crosstab(test_pred_lap, email_test.type)

#Training Data Accuracy (with Laplace smoothing)

train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == email_train.type)
accuracy_train_lap

'''
High training accuracy but low test accuracy: This often
indicates overfitting, where the model has memorized the training data.
Similar training and test accuracy: If both accuracies
are high and similar, this suggests the model is well-fitted
and generalizes well.
Improvement with Laplace Smoothing: If test accuracy improves
after applying Laplace smoothing, it indicates the model
has become more generalizable and performs better
on unseen or rare words in the test data.
Example:
Test Accuracy without Smoothing (accuracy_test_m): 90%
Training Accuracy without Smoothing (accuracy_train_m): 98%
This could indicate overfitting.
Test Accuracy with Smoothing (accuracy_test_lap): 92%
Training Accuracy with Smoothing (accuracy_train_lap): 96%
This suggests the model has become slightly less fitted to 
the training databut perform  better on the test set,
indicating improved generalization.
'''















