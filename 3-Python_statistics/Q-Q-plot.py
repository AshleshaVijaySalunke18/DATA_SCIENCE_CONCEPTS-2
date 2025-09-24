# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:38:59 2025

@author: Ashlesha
"""
#q-q plot
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
measurements = np.random.normal(loc=20, scale=5, size=100)
stats.probplot(measurements, dist="norm", plot=plt)
plt.show()
###########################
#25/4/25
#log transform in machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create the DataFrame
data = {
    'person name': ['Rob', 'Tom', 'Xi', 'mohan', 'pooja', 'sofiya'],
    'credit score': [750, 310, 475, 600, 820, 780],
    'income': [24454, 67656, 55778, 4354, 67545, 50123],  # Added 6th income value
    'age': [32, 23, 34, 56, 43, 34],
    'loan approved?': ['y', 'n', 'y', 'n', 'y', 'n']
}
df = pd.DataFrame(data)

# Apply log transformation to income
df['log_income'] = np.log(df['income'])

# Plotting
plt.figure(figsize=(10, 6))  # fixed typo: was plt.Figure()

# Bar chart: original income
plt.bar(df['person name'], df['income'], color='skyblue', label='Original Income')

# Bar chart: log-transformed income (scaled for visibility)
plt.bar(df['person name'], df['log_income'] * 10000, color='orange', alpha=0.7, label='Log(Income) * 10000')

# Fix typos in axis labeling
plt.xlabel('Person Name')
plt.ylabel('Income')
plt.title('Original vs Log-Transformed Income')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()  # fixed typo: was plt.tight_loyout()
plt.show()

###########################################
#chebyshev-inequality
'''use case: salaries
SD=10,000
we want to konw:
    what percentage of data lies between $10,000 and $70,000?
that range is  meu+-3sigma
meu+-3sigma , so we'll used chebyshev-inequality''' 
def chebyshev_inequality(mu,sigma,lower_bound,upper_bound):
    #calculater no of SD(k)
    k=min(abs(lower_bound-mu),abs(upper_bound-mu))/sigma
   
    #apply chebyshev-inequality
    probability = 1 - (1 / (k ** 2))
#format  the result
    return round (probability*100,2),k
'''
start with your range(e.g from 10k to 70k salaries):
lower_bound=10
upper_bound=70
konw your avarage and SD:
mu=40(mean)
sigma=10(SD)
find how far each bound is from the mean:
distance_from_lower=|lower_bound-mu|=|10-40|=30
distance_from_upper=|upper_bound-mu|=|70-40|=30
take the smaller of these 2didsatance
this keep the interval symmetric ariund the mean(safe zone)
min_distance=min(distance_fron_lower,distance_from_upper)
convert that distance into no of SD:
    k=min_distance/sigma=30/10=3
          
'''
 
'''
if probability=0.8888,
the probability*100=88.88
then round(88.88,2)ensures it's round to 2
decimal place,like 88.89

'''
#input from slide
mean_salary=40000
SD_salary=10000
lower=10000
upper=700000

percent,k_val=chebyshev_inequality(mean_salary,SD_salary,lower,upper)
print(f"According to chebyshev inequality :")
print(f"At least {percent} % of salaries lie between ${lower} and ${upper}")
print(f"(This range is +-{int(k_val)} Stander deviation from the mean)")
######################################################################################
###5-5-25
######################################################################################





















