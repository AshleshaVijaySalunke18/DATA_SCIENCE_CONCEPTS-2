# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 16:28:18 2025

@author: Ashlesha
"""

# ------------------------------------------------------------------
# 1.  TF-IDF on a toy corpus
# ------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Thor eating pizza, Loki is eating pizza, Ironman ate pizza already",
    "Apple is announcing new iphone tomorrow",
    "Tesla is announcing new model-3 tomorrow",
    "Google is announcing new pixel-6 tomorrow",
    "Microsoft is announcing new surface tomorrow",
    "Amazon is announcing new eco-dot tomorrow",
    "I am eating biryani and you are eating grapes",
]

# Create and fit the vectorizer, then transform the corpus
v = TfidfVectorizer()
v.fit(corpus)
transform_output = v.transform(corpus)

# Print the learned vocabulary
# This retrieves every word (feature) that TfidfVectorizer
# picked up during .fit().
print(v.vocabulary_)

# ------------------------------------------------------------------
# Print IDF score of every word
# ------------------------------------------------------------------
all_feature_names = v.get_feature_names_out()

for word in all_feature_names:
    # Get the index of the word in the vocabulary
    indx = v.vocabulary_.get(word)
    # Retrieve the IDF score
    idf_score = v.idf_[indx]
    print(f"{word} : {idf_score}")

# ------------------------------------------------------------------
# 2.  Read an e-commerce dataset
# ------------------------------------------------------------------
import pandas as pd

# Read the data into a pandas DataFrame
df = pd.read_csv("c:/Data-Science/8-RecommendationSystem/Ecommerce_data.csv")
print(df.shape)
df.head(5)

# Check the distribution of labels
df["label"].value_counts()

# Add a numeric label column
df["Label_num"] = df["label"].map(
    {
        "Household": 0,
        "Books": 1,
        "Electronics": 2,
        "Clothing & Accessories": 3,
    }
)

# ------------------------------------------------------------------
# 3.  Train / test split
# ------------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.Text,
    df.Label_num,
    test_size=0.20,      # 20 % of the samples go to the test set
    random_state=2022,   # keeps the split reproducible
    stratify=df.Label_num,
)

"""random_state
 Purpose : Controls the random shuffling
 applied before the split.
Why use it : To make your split reproducible.
 How to choose it : You can use any integer.
 Same number → same result every time.
 Example      : random_state = 2022 ensures that every time      
       you run the code, you get the same train-test split.
 
    
 stratify
 Purpose : Ensures the class distribution in y
  is preserved in both train and test sets.
 Why use it : Important for classification tasks,
   especially with imbalanced classes.
 How to use it : Set it to the same variable as your label
 (stratify = df.Label_num).
 This guarantees each class appears in train and test sets
 in proportion to their frequency in the original dataset.

Why use a specific random_state number?
Pick any integer (e.g., 0, 1, 42, 2022).
Every time you run the code with the same random_state,
you get the same shuffled split.
   
If someone else runs your code, they get the exact same result,
which is critical for debugging and collaboration.
  
It does not matter what number you pick, as long as you
use the same one when you want identical results.

If you omit random_state, the split will be different every run
the code-which is often undesirable in ML experiments.

with randam_state=42
Train:['Sample 4','Sample 9','sample 3',
       'sample 8','sample 6','sample 2','sample5']
Test:['Sample1','sample 8','sample 5']

This shows how setting random_state ensure
consistent splits -crittical for debugging,
reproducibility , and  experiment tracking.

"""
print("Shape of X_train:",X_train.shape)
print("Shape of X_train:",X_test.shape)
y_train.value_counts()
y_test.value_counts()
#############################

#Apply to classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

#1 create a pipeline object
clf=Pipeline([
    ('vectorizer_tfidf',TfidfVectorizer()),
    ('KNN',KNeighborsClassifier())
])

#2 fit with x_train and y_train
clf.fit(X_train,y_train)

#3 get the  prediction for x_ttest and store in y_pred
y_pred=clf.predict(X_test)

#4 print the classfication report
print(classification_report(y_test, y_pred))



