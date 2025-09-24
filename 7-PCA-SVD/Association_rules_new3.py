# Importing necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Step 1: Simulate e-commerce transactions (products purchased per customer)
transactions = [
    ['Laptop', 'Mouse', 'Keyboard'],
    ['Smartphone', 'Headphones'],
    ['Laptop', 'Mouse', 'Headphones'],
    ['Smartphone', 'Charger', 'Phone Case'],
    ['Laptop', 'Mouse', 'Monitor'],
    ['Headphones', 'Smartwatch'],
    ['Laptop', 'Keyboard', 'Monitor'],
    ['Smartphone', 'Charger', 'Phone Case', 'Screen Protector'],
    ['Mouse', 'Keyboard', 'Monitor'],
    ['Smartphone', 'Headphones', 'Smartwatch']
]

# Step 2: Convert the transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Step 3: Apply the Apriori algorithm to find frequent product combinations with reduced support (0.2)
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Step 4: Generate association rules using reduced confidence threshold (0.5)
rules = association_rules(frequent_itemsets, metric="confidence",
                          min_threshold=0.5)

# Step 5: Display the frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)
print("\nAssociation Rules:")
df2=rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
