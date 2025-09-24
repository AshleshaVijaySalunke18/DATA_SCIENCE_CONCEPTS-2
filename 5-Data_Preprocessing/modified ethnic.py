import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data setup
data = {
    'person name': ['Rob', 'Tom', 'Xi', 'Mohin'],
    'credit score': [234, 234, 233, 455],
    'income': [344444, 3400040, 450653, 40593],
    'age': [23, 45, 32, 56],
    'loan approved?': ['Y', 'N', 'Y', 'N']
}
df = pd.DataFrame(data)

# Apply log transformation to income
df['log_income'] = np.log(df['income'])

# ----- Bar chart: Original Income -----
plt.figure(figsize=(10, 5))
plt.bar(df['person name'], df['income'], color='skyblue')
plt.title('Original Income of Individuals')
plt.xlabel('Person Name')
plt.ylabel('Income')
plt.show()

# ----- Bar chart: Log-Transformed Income -----
plt.figure(figsize=(10, 5))
plt.bar(df['person name'], df['log_income'], color='lightgreen')
plt.title('Log-Transformed Income of Individuals')
plt.xlabel('Person Name')
plt.ylabel('Log(Income)')
plt.show()
