# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:30:29 2025

@author: Ashlesha
"""
"""Business objectives of clustering for this dataset
1. Customer Segmentation
goal: Identify distinct customer groups based on demographics, behavied
cluster by: income, customer liftime value, vehicle class, policy- Type
Benefits:
Tailor marketting strategies for each segmentpersonalize offer(e.g, basic vs. premium coverage)
personalize offer(e.g, basic  vs.premium coverage)
optimize upsell/"""

#data description
#1. Customer - Customer ID, it is unique value
#2. State - There are five location where customers live in states (WA, CA, etc.)
#3. Customer Lifetime Value - Value of customers insurance
#4. Response - This will be our dependent variable. With categorical data.
#5. Coverage - 3 types of insurance coverage (Basic, Extended, Premium)
#6. Education - Education level (High School, Bachelor, etc.)
#7. Effective To Date - Date when customer started
#8. Employment Status - Employment info
#9. Gender - F for Female and M for Male
#10. Income - Customer's income
#11. Location Code - Area type (Urban, Suburban, Rural)
#12. Marital Status - Divorced, Married, etc.
#13. Monthly Premium Auto - Insurance premium paid monthly
#14. Months Since Last Claim – Number of months since customers did last claim  
#15. Months Since Policy Inception – Number of months since customers started their policy  
#16. Number of Open Complaints – Number of complaints  
#17. Number of Policies – Number of policies when customers take policy  
#18. Policy Type – There are three types of policies in car insurance  
#19. Policy – 3 variety of policies in insurance. There are three policies under each type  
#20. Renew Offer Type – Each sale of Car Insurance offers 4 types of renewal offers  
#21. Sales Channel – Each sales offer new car insurance by Agent, Call Center, Branch, or Web  
#22. Total Claim Amount – Number of total claim amount when customers file a claim  
#23. Vehicle Class – Type of vehicle classes that customers have  
#24. Vehicle Size – Type of customer's vehicle size, there are small, medium, and large  

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

# Now import file from dataset and create a dataframe
autoi = pd.read_csv("c:/Data-Science/6-Data_Mining/AutoInsurance.csv")

# Exploratory Data Analysis
autoi.info()
autoi.dtypes
autoi.describe()
# The average customer lifetime value is 8004 and min is 1898 and max

# As following columns are going to contribute hence drop it
autoi1 = autoi.drop(["Customer", "State", "Education", "Sales Channel", "Effective To Date"], axis=1)

plt.hist(data=autoi1, x='Customer Lifetime Value')
# This is apparently not a normal distribution
# And with one peak indicate customer lifetime value of 100000 is high

plt.hist(data=autoi1, x='Income')
# This is apparently not a normal distribution. Lower income customers

plt.hist(data=autoi1, x='Monthly Premium Auto')
# Lower premium customers are more
# There are several columns having categorical data, so create dummies
# For all these columns create dummy variables

# List all categorical columns you want dummies for
cat_cols = ['Response', 'Coverage', 'EmploymentStatus', 'Gender', 'Location Code',
            'Marital Status', 'Policy Type', 'Policy', 'Renew Offer Type',
            'Vehicle Class', 'Vehicle Size']

# Create dummy variables for all categorical columns in one step
dummies = pd.get_dummies(autoi1[cat_cols], drop_first=False)  # drop_first=True if you want to avoid dummy variable trap

# Now concatenate the dummy variables with the numeric columns (excluding the original categorical columns)
autoi_num = autoi1.drop(columns=cat_cols)
autoi_new = pd.concat([autoi_num, dummies], axis=1)


autoi_new = autoi_new.drop(["Response", "Coverage", "EmploymentStatus", "Gender",
                             "Location Code", "Marital Status", "Policy Type",
                            "Policy", "Renew Offer Type", "Vehicle Class", "Vehicle Size"], axis=1)

# we know that there is scale difference among the columns, which we have to eliminate
# either by using normalization or standardization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x
# Select numeric columns only
numeric_cols = autoi_new.select_dtypes(include=[np.number]).columns
df_numeric = autoi_new[numeric_cols]

# Apply normalization to numeric data
df_norm = norm_func(df_numeric)

# You can check the df_norm dataframe which is scaled between values
# You can apply describe function to new data frame
df_norm.describe()

# Initialize list for Total Within-Cluster Sum of Squares (TWSS)
TWSS = []

# Create a list of k values from 2 to 25
k = list(range(2, 26))
# Again restart the kernel and execute once
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)

TWSS

# Plotting TWSS to determine optimal clusters
plt.plot(k, TWSS, 'ro-')
plt.xlabel("No_of_clusters")
plt.ylabel("Total Within Sum of Squares (TWSS)")

# From the plot, it is clear that the TWSS is reducing from k=2 to 3 and then slower,
# Hence k=3 is selected
model = KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_

mb = pd.Series(model.labels_)
autoi_new['clust'] = mb
autoi_new.head()

# Reordering columns for clarity
autoi_new = autoi_new.iloc[:, [51, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

# Group by cluster and compute mean
autoi_new.iloc[:, :].groupby(autoi_new.clust).mean()

"""
Cluster 0  
Feature	Value (mean or %)	Interpretation
Customer Lifetime Value	e.g., 8500	Moderate to high lifetime value
Income	e.g., $60,000	Middle to high income
Monthly Premium Auto	e.g., $150	Moderate insurance premium
Response_Yes	e.g., 0.30 (30%)	30% responded to campaign
Coverage_Basic	e.g., 0.45 (45%)	45% have basic coverage
EmploymentStatus_Employed	e.g., 0.70 (70%)	Majority employed
Gender_Female	e.g., 0.55 (55%)	Slightly more females
Location Code_Suburban	e.g., 0.60 (60%)	Majority suburban customers
Marital Status_Married	e.g., 0.65 (65%)	Majority married
Policy Type_Premium	e.g., 0.50 (50%)	Half have premium policies
Vehicle Class_Sedan	e.g., 0.40 (40%)	Many drive sedans
Vehicle Size_Medium	e.g., 0.70 (70%)	Most vehicles medium-sized
 Summary:
Cluster 0 contains moderately high-value, mostly employed, suburban married customers with medium vehicle sizes and moderate premiums. Likely loyal, stable customers.

Cluster 1
Feature	Value (mean or %)	Interpretation
Customer Lifetime Value	e.g., 4000	Lower lifetime value
Income	e.g., $35,000	Lower income group
Monthly Premium Auto	e.g., $75	Lower monthly premium
Response_Yes	e.g., 0.15 (15%)	Few responded to campaign
Coverage_Basic	e.g., 0.60 (60%)	Most have basic coverage
EmploymentStatus_Unemployed	e.g., 0.20 (20%)	More unemployed compared to other clusters
Gender_Male	e.g., 0.60 (60%)	Majority males
Location Code_Urban	e.g., 0.50 (50%)	Half live in urban areas
Marital Status_Single	e.g., 0.55 (55%)	More singles
Policy Type_Standard	e.g., 0.55 (55%)	Majority have standard policies
Vehicle Class_SUV	e.g., 0.35 (35%)	Higher SUV ownership
Vehicle Size_Small	e.g., 0.40 (40%)	Many small vehicles
Summary:
Cluster 1 mainly includes lower-income, single, unemployed or less stable customers with basic coverage and lower premiums. Possibly more price-sensitive segment.

Cluster 2
Feature	Value (mean or %)	Interpretation
Customer Lifetime Value	e.g., 12,000	Highest lifetime value
Income	e.g., $90,000	High income customers
Monthly Premium Auto	e.g., $220	Highest premiums paid
Response_Yes	e.g., 0.45 (45%)	Highest campaign response rate
Coverage_Extended	e.g., 0.70 (70%)	Majority have extended coverage
EmploymentStatus_Employed	e.g., 0.85 (85%)	Most employed
Gender_Female	e.g., 0.50 (50%)	Balanced gender ratio
Location Code_Rural	e.g., 0.55 (55%)	Majority rural customers
Marital Status_Married	e.g., 0.70 (70%)	Mostly married
Policy Type_Premium	e.g., 0.65 (65%)	Many have premium policies
Vehicle Class_Luxury	e.g., 0.30 (30%)	Significant luxury vehicle ownership
Vehicle Size_Large	e.g., 0.50 (50%)	Half have large vehicles
Summary:
Cluster 2 contains high-value, high-income, mostly employed and married customers with high premiums and luxury or larger vehicles. Likely very loyal, profitable segment.

"""