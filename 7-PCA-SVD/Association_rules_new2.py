# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:20:20 2024

@author: Dell
"""

# Importing necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 1: Simulating healthcare transactions (symptoms/diseases/treatments per patient)
healthcare_data = [
    ['Fever', 'Cough', 'COVID-19'],
    ['Cough', 'Sore Throat', 'Flu'],
    ['Fever', 'Cough', 'Shortness of Breath', 'COVID-19'],
    ['Cough', 'Sore Throat', 'Flu', 'Headache'],
    ['Fever', 'Body Ache', 'Flu'],
    ['Fever', 'Cough', 'COVID-19', 'Shortness of Breath'],
    ['Sore Throat', 'Headache', 'Cough'],
    ['Body Ache', 'Fatigue', 'Flu'],
]

# Step 2: Convert the data into a format suitable for Apriori using TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(healthcare_data).transform(healthcare_data)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Step 3: Apply the Apriori algorithm to find frequent itemsets with a minimum support of 0.3
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
#apriori: The Apriori algorithm is used to find frequent itemsets 
#with a support threshold of 0.3 (i.e., patterns that occur in at least 30% of the patient records).
# Step 4: Generate association rules using confidence as the metric with a threshold of 0.7
rules = association_rules(frequent_itemsets, metric="confidence",
                          min_threshold=0.7)
#association_rules: We generate association rules from 
#the frequent itemsets with a confidence threshold of 0.7. 
#This ensures that only rules with a confidence of 70% or higher (i.e., strong relationships) are selected.
# Step 5: Display the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
