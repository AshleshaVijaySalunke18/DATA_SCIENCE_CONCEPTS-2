# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:37:13 2025

@author: Ashlesha
"""

'''
We are analyzing a dataset of SMS messages labeled as
"ham" (not spam) or "spam". The goal is to understand
the structure, distribution, and content of messages
to prepare them for a machine learning pipeline using
text classification.
'''

import pandas as pd
df = pd.read_csv("c:/Data-Science/9-Classification/sms_raw_NB.csv", encoding="ISO-8859-1")
# Basic Information
print(df.shape)
df.info()
df.head()

'''
Total Rows: 5,559
Total Column: 2
type: class label(ham or spam)
text:SMS message body
'''

#missing  values check
df.isnull().sum()
#no missing values.Dataset is complete and doesn't need imputation
#class Distribution

df['type'].value_counts()
'''
| Class | Count m| Percentage |
|-------|--------|------------|
| ham   | 4812   | 86.6%      |
| spam  | 747    | 13.4%      |
'''
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='type', data=df)
plt.title("Distribution of Ham and Spam Messages")
plt.show()

# Understand the Distribution

# Create a new column for message length
df['text_Length'] = df['text'].apply(len)

# Set Seaborn style
sns.set(style="whitegrid")

# Plot 1: Class distribution (bar + pie)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot for counts
sns.countplot(x='type', data=df, ax=ax[0], palette="Set2")
ax[0].set_title("Class Distribution: Ham vs Spam")
ax[0].set_xlabel("Message Type")
ax[0].set_ylabel("Count")

#pie chat for proportion
class_counts=df['type'].value_counts()
ax[1].pie(
    class_counts,
    labels=class_counts.index,
    autopct='%1.1f%%',
    colors=["skyblue","salmon"],
    startangle=90,
    wedgeprops={'edgecolor':'black'})
ax[1].set_title("Preoportion of Ham and Spam Messange")

plt.tight_layout()
plt.show()

'''
Imbalanced Dataset: A majority of messages are ham.
This will influence model evaluation –
we must use metrics like precision, recall, and F1-score
in addition to accuracy.
'''

# Text Length Analysis
df['text_length'] = df['text'].apply(len)
df[['type', 'text_length']].groupby('type').describe()

sns.histplot(data=df, x='text_length', hue='type', bins=50, kde=True)
plt.title("Message Length Distribution by Type")
plt.xlabel("Number of Characters")
plt.ylabel("Message Count")
plt.show()

'''
Observation : spam messages tend to be longer than ham
messange. this may be because spam contains
full advertisements.URLs,or legal notices.
'''

# Word Count Per Message
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df[['type', 'word_count']].groupby('type').describe()

# Longer word count is often associated with spam.
# Models can benefit from this as a numeric feature.

# Most Frequent Messages
df['text'].value_counts().head()
'''
| Message                          | Count | Label |
|----------------------------------|-------|-------|
| Sorry, I’ll call later           | 30    | ham   |
| I am in meeting, call me later   | 21    | ham   |
| Where are you?                   | 16    | ham   |

'''
'''
Repetitive ham messages suggest casual language.
'''
# Character-Level Cleaning (Preliminary Step)
import re

def clean_text(text):
    return " ".join([word.lower() for word in re.sub("[^a-zA-Z]", " ", text).split() if len(word)>3])

df['clean_text'] = df['text'].apply(clean_text)
df[['text', 'clean_text']].head()

# Most Common Words (Preview)
from collections import Counter

all_words = ' '.join(df['clean_text']).split()
word_freq = Counter(all_words)
pd.DataFrame(word_freq.most_common(20), columns=["Word", "Frequency"])


#plot 2 message length distribution by message type
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='text_length', hue='type', bins=50, kde=True)
plt.title("Distribution of Message Length by Message Type")
plt.xlabel("Message Length (characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
 #########################################

#data preprocessing

import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# For SMOTE
from imblearn.over_sampling import SMOTE

# --- 1. Clean Text ---
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)  # Remove non-letters
    words = [word.lower() for word in text.split() if len(word) > 3]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

#remove empty cleaned messages
df=df[df['clean_text'].str.strip()!='']

# --- 2. Encode Target Labels ---
le = LabelEncoder()
df['label'] = le.fit_transform(df['type'])  # ham=0, spam=1

# --- 3. Feature Extraction (BoW + TF-IDF) ---
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(df['clean_text'])

tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_counts)

# Features and Labels
X = X_tfidf
y = df['label']

# --- 4. Train-Test Split ---
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 5. Apply SMOTE ---
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# --- 6. Check Class Balance After SMOTE ---
from collections import Counter

print("Before SMOTE:", Counter(y_train))
print("After SMOTE :", Counter(y_train_sm))


'''
Why SMOTH is useful here:
it  synthetically creates new minority class(spam)examples.
prevents the model from being biased toward the majority
class(ham) Essential when using  standerd classificers like naive boyes
or SVM
'''

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# model with Laplace smoothing (alpha=1.0 by default)
nb_model_default = MultinomialNB(alpha=1.0)
nb_model_default.fit(X_train_sm, y_train_sm)

# Predictions
y_pred_test_default = nb_model_default.predict(X_test)
y_pred_train_default = nb_model_default.predict(X_train_sm)

# Evaluation
from sklearn.metrics import accuracy_score

print(" Without Laplace Smoothing")
print("Train Accuracy:", accuracy_score(y_train_sm, y_pred_train_default))
print("Test Accuracy :", accuracy_score(y_test, y_pred_test_default))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_pred_test_default))
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred_test_default))

# Model With Laplace Smoothing (e.g., alpha = 3)
nb_model_laplace = MultinomialNB(alpha=3)
nb_model_laplace.fit(X_train_sm, y_train_sm)

# Predictions

y_pred_test_lap=nb_model_laplace.predict(X_test)
y_pred_train_lap=nb_model_laplace.predict(X_train_sm)

#Evalution

print("\nWith Laplace Smoothing (Alpha=3)")
print("Train Accuracy:", accuracy_score(y_train_sm, y_pred_train_lap))
print("Test Accuracy :", accuracy_score(y_test, y_pred_test_lap))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_pred_test_lap))
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred_test_lap))









