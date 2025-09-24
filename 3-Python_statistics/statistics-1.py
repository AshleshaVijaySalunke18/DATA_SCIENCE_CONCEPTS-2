# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:59:25 2025

@author: Ashlesha
"""

import pandas as pd
import numpy as np
df = pd.read_csv("C:/Data-Science/3-Python_statistics/income.csv", names=["name", "income"], skiprows=[0])
df 
df.income.describe()
df.income.quantile(0)
df.income.quantile(0.25, interpolation="higher")
df.income.quantile(0.75)
df.income.quantile(1)
df.income.quantile(0.99)
percentile_99=df.income.quantile(99)
df
df['income'][3]=np.NaN
df
df.income.mean()
df_new=df.fillna(df.income.mean())
df_new
df_new=df.fillna(df.income.median())
df_new



######################################
###----23-4-25-----
import numpy as np
data=[10,12,23,23,26,23,21,16]
print("Original Data:",data)
#step1:mean of the data
mean=np.mean(data)
print("\nMean(Avarage):",mean)
#step2: mean absolute Derivation(MAD)
#it's the avarage of absolute derivation from the mean
mad=np.mean([abs(x-mean)for x in data])
print("Mean Absolute Derivation(MAD):",mad)
#step 3 Variance
#it is the avarage of square differences from the mean
Variance=np.var(data)
print("Variance:",Variance)
#step4 standerd derivation
std_dev=np.std(data)#by defult,population stander derviation
print("Standerd Derivation:",std_dev)
######################################
#History of math test scores from the images
history_scores=[75,72,68,67,73]
math_scores=[93,96,43,47,51,90]
def calculate_mad(scores):
    mean=sum(scores)/len(scores)
    mad=sum(abs(x-mean)for x in scores)/len(scores)
    return mad

#calcluat MAD
history_mad=calculate_mad(history_scores)
math_mad=calculate_mad(math_scores)
print("Mean Absolute Derivation(MAD):")
print(f"History Test:{history_mad:.2f}")
print(f"Math Test: {math_mad:.2f}") 
'''
Although both the history test and the math test
have the same avarage score of 70,their mean absolute deviation(MAD)
tell a different story.
The MAD of history test is lower,which means the scores are closes to mean .
the students perform more consistenly

'''
######################################
import math

# Dataset 1: from left table
scores_1 = [75, 72, 68, 65, 67, 73]

# Dataset 2: from right table
scores_2 = [83, 70, 70, 63, 70, 70]

# Correct function name
def calculate_standard_deviation(scores):
    mean = sum(scores) / len(scores)
    square_diffs = [(x - mean) ** 2 for x in scores]
    variance = sum(square_diffs) / len(scores)
    std_dev = math.sqrt(variance)
    return std_dev

# Calculate standard deviations
std_dev_1 = calculate_standard_deviation(scores_1)
std_dev_2 = calculate_standard_deviation(scores_2)

print(f"Standard Deviation of Scores 1: {std_dev_1:.2f}")
print(f"Standard Deviation of Scores 2: {std_dev_2:.2f}")
'''
Even though both dataset have the same avarage(mean)
dataset1(left table)
standard deviation=3.55
most scores are colse to the mean
(small squared differences).
indicates consistent performance
among students with low varibility.

dataset2(right table)
standard deviaation=6.02
some score (like 82 and 63)are far from  the mean,
causing larger square different
indicates higher variability

'''
######################################
import numpy as np
#original weights
original_weights=[105,156,145,172,100]
#add 5 pounds for winter clothing
adjusted_weights=[weight+5 for weight in original_weights]
#calculate mean and stander deviation
mean_original = np.mean(original_weights)
std_original = np.std(original_weights, ddof=1)

mean_adjusted = np.mean(adjusted_weights)
std_adjusted = np.std(adjusted_weights, ddof=1)
'''
ddof=1 in np.std()
ddof 

'''
print("Original Mean:", mean_original)
print("Original Std Dev:", std_original)
print("Adjusted Mean:", mean_adjusted)
print("Adjusted Std Dev:", std_adjusted)
######################################
import numpy as np
#original Data
weights=[102,345,56,764,675]
#water fourmula:(weight *2.5)+100
water_intake=[(w*2.5)+100 for w in weights]
#claculate original stats
mean_weight=np.mean(weights)
std_weight=np.std(weights,ddof=1)
#calculate new start
mean_water=mean_weight*2.5+100
std_water=std_weight*2.5
print(f"Original mean weight:{mean_weight:2f},original Std Dev:{std_weight:.2f}")
print(f"mean water intake:{mean_water:.2f}ml")
print(f"Std Dev of water intake: {std_water:.2f} ml")




######################################


 