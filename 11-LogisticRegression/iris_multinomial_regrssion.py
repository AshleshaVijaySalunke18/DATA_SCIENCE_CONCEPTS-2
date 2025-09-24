# -*- coding: utf-8 -*-
"""
Created on  21-8- 2025

@author: Ashlesha
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ------------------------------------
# 1. Load dataset
# ------------------------------------
iris = load_iris()

# Features (X) = flower measurements
# Analogy: Hours spent on different activities (football, cricket, tennis practice)
X = iris.data

# Target (y) = flower species (Setosa, Versicolor, Virginica)
# Analogy: Which sport is chosen (Football, Cricket, Tennis)
y = iris.target

# ------------------------------------
# 2. Split into Train & Test
# ------------------------------------
# Train = learning phase (students practicing sports)
# Test = exam phase (checking if model correctly predicts the sport)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------------
# 3. Train Multinomial Logistic Regression
# ------------------------------------
# Model learns the "softmax equation"
# Analogy: Model learns how hours spent on football/cricket/tennis → probability of sport
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
# In scikit-learn’s LogisticRegression, the parameter solver='lbfgs'
# specifies that the model will use the L-BFGS optimization algorithm to estimate the best weights

'''
lbfgs = Limited-memory BFGS → a smart algorithm that finds the best coefficients faster than plain gradient descent.
It is memory-efficient and works well when we have many features.
It is recommended for multinomial logistic regression.

Other solvers:
newton-cg – also works for multinomial regression.
saga, sag – good for very large datasets.
'''

model.fit(X_train, y_train)

# 4. Make Predictions
# -------------------------------
# Model predicts probabilities for each class (species/sport)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 5. Evaluate Performance
# -------------------------------
# Prints precision, recall, F1-score for each class
# Analogy: How well does the model identify "football players" vs "cricket players"
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
'''
Classification Report:
               precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        19
  versicolor       1.00      1.00      1.00        13
   virginica       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45


'''
# Class-wise accuracy (per flower type)
print("\nClass-wise Accuracy:")
for i, class_name in enumerate(iris.target_names):
    class_indices = (y_test == i)  # where the true class is i
    correct = np.sum(y_pred[class_indices] == y_test[class_indices])
    total = np.sum(class_indices)
    acc = correct / total
    print(f"{class_name}: {acc:.2f}")

'''
Class-wise Accuracy:
setosa: 1.00
versicolor: 1.00
virginica: 1.00
'''

# -------------------------------
# The reason you're getting accuracy = 1.0 for all classes is that your test set happened...



'''
The reason you're getting accuracy = 1.0 for all classes is
that your test set happened to be perfectly predicted by
the logistic regression model. This is possible with the Iris dataset,
since it's small, clean, and very well-separated.

But in general, you won't always get perfect accuracy --
it depends on the random split (random_state) and the model performance.

Let me explain clearly:

Why Accuracy = 1 for All Classes?
The Iris dataset is linearly separable,
especially Setosa (always predicted correctly).

Logistic regression with lbfgs solver converges very well.

With the 70-30 split,
the test set was easy enough that the model
got everything right.

'''


# 6. Example Predictions with Explanation
# -------------------------------
print("\nSample Predictions with Probabilities:")
for i in range(5):  # first 5 test samples
    true_class = iris.target_names[y_test[i]]
    predicted_class = iris.target_names[y_pred[i]]
    probs = y_proba[i]
    print(f"Sample {i+1}: True = {true_class}, Predicted = {predicted_class}, F = {probs}")
'''
Sample Predictions with Probabilities:
Sample 1: True = versicolor, Predicted = versicolor, F = [0.00409969 0.81234384 0.18355648]
Sample 2: True = setosa, Predicted = setosa, F = [9.41955388e-01 5.80440320e-02 5.80205686e-07]
Sample 3: True = virginica, Predicted = virginica, F = [1.58411406e-08 2.09129639e-03 9.97908688e-01]
Sample 4: True = versicolor, Predicted = versicolor, F = [0.0068249  0.77325142 0.21992368]
Sample 5: True = versicolor, Predicted = versicolor, F = [0.00159401 0.751206   0.2472    ]

'''
# analogy (like football choice)
print(f"Model says: {predicted_class} is the right choice "
      f"highest probability {max(probs)*100:.1f}%.\n")

#Model says: versicolor is the right choice highest probability 75.1%.