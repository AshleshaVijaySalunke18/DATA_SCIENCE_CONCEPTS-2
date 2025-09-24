import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset (update path if needed)
df = pd.read_csv('c:/Data-Science/2-Python_DS/Salary_Data.csv')

# Rename columns for clarity
df.columns = ['Gender', 'Age', 'Year of Experience', 'Salary']

# Display basic statistics and head
df.head()

# KDE plots for continuous variables
sns.displot(df['Age'], kde=True)
sns.displot(df['Year of Experience'], kde=True)
sns.displot(df['Salary'], kde=True)

# Scatter plots (jointplots)
sns.jointplot(x='Age', y='Salary', data=df)
sns.jointplot(x='Age', y='Year of Experience', data=df, kind='reg')  # Regression line plot
sns.jointplot(x='Year of Experience', y='Salary', data=df, kind='hex')  # Hexbin plot

# Pairp between variables
sns.pairplot(df)

# Categorical data visualization
# For gender distribution
sns.countplot(x='Gender', data=df)

# Pie chart and bar chart for Gender count
df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.show()

df['Gender'].value_counts().plot(kind='bar', color='lightblue')
plt.title('Gender Count')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Boxplots (checking for outliers)
sns.boxplot(x='Gender', y='Salary', data=df)
sns.boxplot(x='Gender', y='Age', data=df)
sns.boxplot(x='Gender', y='YearExp', data=df)

# Heatmap to understand correlation coefficients between numerical columns
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Scatter plot - Age vs. Salary
plt.scatter(df['Age'], df['Salary'], color='purple', alpha=0.8)
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# Boxplot for Salary distribution
plt.boxplot(df['Salary'])
plt.title('Salary Distribution')
plt.show()

# Count of Gender (Male vs. Female)
df['Gender'].value_counts().plot(kind='bar', color='orange')
plt.title('Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# FacetGrid – Salary by Age and Gender (Smoker/Non-Smoker as an example)
fg = sns.FacetGrid(df, row='Gender')
fg.map(sns.histplot, 'Salary')
