# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 09:13:11 2025

@author: Ashlesha
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('c:/Data-Science/2-Python_DS/Mall_Customers.csv')
df.head()
df.columns=['cust_id','Gender','Age','Anual_income','Spend_score']
sns.displot(df.Age, kde=True)
sns.displot(df.Anual_income,kde=True)
sns.displot(df.Spend_score,kde=True)
df.dtypes
sns.jointplot(x=df.Age, y=df.Spend_score)
#maximum spending occurs dusring 20-30 , even at 50 too
sns.jointplot(x=df.Age,y=df.Spend_score,kind='reg')
#spending reduces as age increasaes
sns.jointplot(x=df.Age,y=df.Spend_score,kind='hex')
sns.pairplot(df[['Age','Anual_income','Spend_score','Gender']])
#catogrical column
sns.countplot(x ='Gender',data=df)
df['Gender'].value_counts().plot(kind='pie')
df['Gender'].value_counts().plot(kind='bar')
# Create income groups for countplot
df['Income_Group'] = pd.cut(df['Anual_income'], bins=[0, 40, 70, 150], labels=['Low', 'Medium', 'High'])
sns.countplot(x='Income_Group', data=df)
# Boxplots (checking for outliers)
sns.boxplot(y='Anual_income', data=df)
sns.boxplot(y='Spend_score', data=df)
sns.boxplot(y='Age', data=df)
# FacetGrid – Spend score by Age and Gender
fg = sns.FacetGrid(df, row='Gender')
fg.map(sns.histplot, 'Spend_score')
# Histogram of Age using Seaborn
sns.histplot(df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
# Histogram of Spend_score using Seaborn
sns.histplot(df['Spend_score'], bins=20, color='blue', edgecolor='black')
plt.title('Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Histogram of Anual_income using matplotlib
plt.hist(df['Anual_income'], bins=20, color='green', edgecolor='black')  # fixed color string
plt.title('Anual_income')
plt.xlabel('Anual_income')
plt.ylabel('Count')
plt.show()

#scatter plot of Anual_income vs Age
plt.scatter(df['Anual_income'], df['Age'], color='purple', alpha=0.8)  
plt.title('Anual_income  vs Age')
plt.xlabel('Anual_income')
plt.ylabel('Age')
plt.show()

#scatter plot of Anual_income vs Spend_score
plt.scatter(df['Anual_income'], df['Spend_score'], color='Pink', alpha=0.8)  
plt.title('Anual_income  vs Spend_score')
plt.xlabel('Anual_income')
plt.ylabel('Spend_score')
plt.show()

#scatter plot of Age vs Spend_score
plt.scatter(df['Age'], df['Spend_score'], color='Red', alpha=0.8)  
plt.title('Age  vs Spend_score')
plt.xlabel('Age')
plt.ylabel('Spend_score')
plt.show()

#simple corrrelation heatmap
corr=df.corr(numeric_only=True)
plt.imshow(corr, cmap='coolwarm',interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)),corr.columns,rotation=45)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Correlation Matrix')
plt.show()

#boxplot
plt.boxplot(df['Age'])
plt.title('boxplot-Age')
plt.show()

plt.boxplot(df['Spend_score'])
plt.title('boxplot-Spend_score')
plt.show()

plt.boxplot(df['Anual_income'])
plt.title('boxplot-Age')
plt.show()

#count of days
age_bins = pd.cut(df['Age'], bins=range(0, 101, 10), right=False)
age_bins.value_counts().sort_index().plot(kind='bar', color='brown')
plt.title('Count by Age Group (in 10s)')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

#count of gender
df['Gender'].value_counts().plot(kind='bar', color='lightblue')
plt.title('Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

#gender pic chart
df['Gender'].value_counts().plot(kind='pie',autopct='%1.1f%%')

