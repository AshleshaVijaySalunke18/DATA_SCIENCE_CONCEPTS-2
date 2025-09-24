# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:52:43 2025

@author: Ashlesha
"""

'''
Hypothesis Testing for Single Population proportion
Problem Statement:
Billing statements in particular hotel has some error.
There was error of 15%.
Out of 1000 billing statements which were randomly selected contains 102 errors.
Use α = 0.10, and determine whether the proportion of billing statement is less than 15%.


'''
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import norm

# ---------------------
# Given data
# ---------------------
x = 102            # Number of errors
n = 1000           # Total sample size
p0 = 0.15          # Hypothesized population proportion
alpha = 0.10       # Significance level

# ---------------------
# Step 1: One-sample Z-test for proportions
# ---------------------
z_stat, p_value = proportions_ztest(count=x, nobs=n, value=p0, alternative='smaller')

# ---------------------
# Step 2: Critical z-value for left-tailed test
# ---------------------
z_critical = norm.ppf(alpha)  # z at 10% significance level for one-tailed test

# ---------------------
# Step 3: Print intermediate results
# ---------------------
print("Sample Proportion (p̂):", round(x/n, 3))
print("Z-statistic:", round(z_stat, 3))
print("P-value:", round(p_value, 4))
print("Critical Z-value:", round(z_critical, 3))

#Sample Proportion (p̂): 0.102
#Z-statistic: -5.015
#P-value: 0.0
#Critical Z-value: -1.282

# ---------------------
# Step 4: Conclusion
# ---------------------
if z_stat < z_critical:
    print("Result: Reject the null hypothesis (p < 0.15)")
else:
    print("Result: Fail to reject the null hypothesis (not enough evidence that p < 0.15)")

# Result: Reject the null hypothesis (p < 0.15)