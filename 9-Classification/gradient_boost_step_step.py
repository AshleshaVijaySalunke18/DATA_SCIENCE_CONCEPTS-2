# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:29:40 2025

@author: Ashlesha
"""

import numpy as np
import matplotlib.pyplot as plt
'''
Step 2: Create Synthetic Dataset Generates 100 random 
values between -0.5 and 0.5.
The target variable y is a non-linear 
function: y = 3x² + noise.
The noise is added with 0.05 * np.random.randn(100)
to simulate real-world data.
'''
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3 * X[:, 0]**2 + 0.05 * np.random.randn(100)
'''
Step 3: Store in DataFrame and Plot Store X and y 
into a DataFrame. Plot the scatterplot of the data
to visualize the relationship.
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

df = pd.DataFrame()
df['x'] = X.reshape(100)
df['y'] = y

plt.scatter(df['x'], df['y'])
plt.title('X vs y')
plt.show()

df['pred1'] = df['y'].mean()
df['res1'] = df['y'] - df['pred1']

plt.scatter(df['x'], df['y'])
plt.plot(df['x'], df['pred1'], color='red')
plt.show()
#step 5 DecisionTreeRegressor
tree1 = DecisionTreeRegressor(max_leaf_nodes=8)
tree1.fit(df['x'].values.reshape(-1, 1), df['res1'].values)

plot_tree(tree1)
plt.show()

'''
Step 6: Visualize Predictions After First Tree
0.265458 is the mean of y, acting as initial prediction.

Add the tree’s predicted residuals to the mean for updated prediction.
np.linspace(start, stop, num) is a NumPy function that
 returns evenly spaced numbers over a specified interval.
What This Line Does  
-0.5 → starting value of the range  
0.5 → ending value of the range  
A 1D array of 500 values evenly spaced from -0.5 to 0.5.  
500 → number of evenly spaced points between -0.5 and 0.5
Why  it's  used here 
This array (X_test) is used to: Evaluate and plot the model’s
predictions at many points across the input space.  
Gives a smooth prediction curve when we draw y_pred vs X_test.  
It’s especially helpful in regression problems to 
visualize how the model behaves between and beyond the training points.

'''
#generating X_test
X_test = np.linspace(-0.5,0.5,500)
y_pred = 0.265458 + tree1.predict(X_test.reshape(500, 1))  # sklearn expects 2 D array
plt.figure(figsize=(14, 4))
plt.subplot(121)
plt.plot(X_test, y_pred, linewidth=2, color='red')
plt.scatter(df['x'], df['y'])

#step 7 new residuals updates
df['pred2']=0.265458 + tree1.predict(df['x'].values.reshape(100,1))
df

df['res2']=df['y']-df['pred2']
df

#trains second tree on new residuals res2
tree2=DecisionTreeRegressor(max_leaf_nodes=8)
tree2.fit(df['x'].values.reshape(100,1),df['res2'].values)
#df['x'].values get numpy array of that column

#total prediction -mean+tree1+tree2 prediction
y_pred = 0.265458 + sum(regressor.predict(X_test.reshape(-1, 1)) for regressor in [tree1, tree2])
#reshape(-1,1)is going reshape 'x'500,1
plt.figure(figsize=(14,4))
plt.subplot(121)
plt.plot(X_test,y_pred,linewidth=2,color='red')
plt.scatter(df['x'],df['y'])
plt.title('x vs y')

#step 9 recusive gradient boosting function

def gradient_boost(X, y, number, lr, count=1, regs=[], foo=None):
    if number == 0:
        return  # Stops recursion when required trees are built.
    else:
        # do gradient boosting

        if count > 1:
            y = y - regs[-1].predict(X)  # Updates residuals (error) after each tree.
            # here y becomes residuals
        else:
            foo = y  # For first iteration, just store original y.

        tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
        tree_reg.fit(X, y)

        regs.append(tree_reg)  # Train a new tree on the current residual.

        x1 = np.linspace(-0.5, 0.5, 500)  # Create test data for plotting
        y_pred = sum(lr * regressor.predict(x1.reshape(-1, 1)) for regressor in regs)
        # Sum predictions from all trees multiplied by learning rate (like real boosting)

        print(number)
        plt.figure()
        plt.plot(x1, y_pred, linewidth=2)
        plt.plot(X[:, 0], foo, "r.")  # Scatter original data in red
        plt.title(f"Boosting Iteration: {count}")
        plt.show()

       #show prediction line after each stage and original data
        gradient_boost(X, y, number - 1, lr, count + 1, regs, foo)
        #recursively call function to build more trees

#final call  to gradient boosting  function

np.random.seed(42)
x=np.random.rand(100,1)-0.5
y=3*x[:,0]**2+0.05*np.random.randn(100)
gradient_boost(X, y, 5, lr=1)
       




