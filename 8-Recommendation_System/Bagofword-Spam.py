# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 16:18:21 2025

@author: Ashlesha
"""

# 1. Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# 2. Load the dataset
# Assumes 'spam.csv' has at least two columns: 'Category' (spam/ham) and 'Message'
df = pd.read_csv("c:/Data-Science/8-RecommendationSystem/spam.csv", encoding='latin-1')
# 'latin-1' handles special characters

# Optional: Keep only the necessary columns
df = df[['Category', 'Message']]

# 3. Create a new binary column: 1 for spam, 0 for ham
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# 4. Split the dataset into features and labels
X_train, X_test, y_train, y_test = train_test_split(
    df['Message'],     # Features: the actual messages
    df['spam'],        # Labels: 1 = spam, 0 = ham
    test_size=0.2,     #20% test data
    random_state=42    #for reproducibility
)
# 5. Create a Bag of Words model using CountVectorizer
vectorizer = CountVectorizer()

# 6. Fit the vectorizer on training data and transform both train and test
X_train_cv = vectorizer.fit_transform(X_train)
X_test_cv = vectorizer.transform(X_test)#not fit transform

# 7. Train a Naive Bayes classifier on the vectorized training data
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# 8. Predict on the test set
y_pred = model.predict(X_test_cv)

# 9. Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

#10. Try  with  new massages
new_messages=[
    '''congratulations! You've won  a free ticket to Bahamas.
    Reply WIN to claim . '''
    "Hey, are wr still meeting for lunch today?"
]

#11. BONUS : Use a Pipeline (Cleaner approch)
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Fit the pipeline directly on raw text and labels
pipeline.fit(X_train, y_train)

# Evaluate the pipeline directly on raw text and labels
y_pred_pipeline = pipeline.predict(X_test)
print("\nClassification Report (Pipeline):")
print(classification_report(y_test, y_pred_pipeline))
