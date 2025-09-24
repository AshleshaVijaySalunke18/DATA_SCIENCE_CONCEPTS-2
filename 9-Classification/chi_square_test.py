# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:37:33 2025

@author: Ashlesha
"""

import pandas as pd
from scipy.stats import chi2_contingency  

# Step 1: Create the contingency table (raw data)

data = [[40, 20],  # Male
        [30, 30]]  # Female

# Optional: Convert to a DataFrame for better formatting and visualization
df = pd.DataFrame(data,columns=["Product A", "Product B"],  index=["Male", "Female"])          

# Step 2: Perform the Chi-Square test
chi2, p, dof, expected = chi2_contingency(df)

# Step 3: Print Chi-Square test results
print("Chi-Square Statistic:", round(chi2, 3))  
print("Degrees of Freedom:", dof)                
print("P-Value:", round(p, 4))                  
print("\nExpected Frequencies:\n", pd.DataFrame(expected, columns=df.columns,index=df.index))  

# Step 4: Final Interpretation based on p-value
if p < 0.05:
    print("Reject the null hypothesis – there is a significant association.")
else:
    print("Fail to reject the null hypothesis – no significant association.")
