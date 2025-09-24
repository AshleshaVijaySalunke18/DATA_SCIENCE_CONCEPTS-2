# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:16:20 2025

@author: Ashlesha
"""

import pandas as pd
import numpy as np

letters = pd.read_csv("c:/Data-Science/9-Classification/SVM/Letterdata.csv")

'''
dataset typically used for handwritten character recognition 
or related machine learning tasks. Here's a breakdown of its structure:

letter: Represents the target class (the letter being identified, e.g., 'A')
Features (xbox to yedgex): These are numeric attributes
describing various geometric or statistical properties of the character.

xbox and ybox: X and Y bounding box dimensions.
width and height: Width and height of the character's bounding box.
onpix: Number of on-pixels in the character's image.
xbar and ybar: Mean X and Y coordinate values of on-pixels.
x2bar, y2bar, and xybar: Variances and covariance of pixel intensities.
x2ybar and xy2bar: Additional statistical metrics
for spatial relationships between pixels.
xedge, xedgex, yedge, and yedgex: Edge-related metrics,
possibly representing transitions or edge density in the character image.
'''

# let us carry out EDA
a = letters.describe()

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train, test = train_test_split(letters, test_size=0.2)

# let us split the data in terms X and y for both train and test data
train_X = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_X = test.iloc[:, 1:]
test_y = test.iloc[:, 0]

# kernel Linear
model_linear = SVC(kernel="linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

# Now let us check the accuracy = 0.85675
np.mean(pred_test_linear == test_y)
#0.85825

# kernel rbf
model_rbf = SVC(kernel="rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

# Now let us check the accuracy = 0.92275
np.mean(pred_test_rbf == test_y)
# 0.93175
