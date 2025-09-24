# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 15:54:05 2025

@author: Ashlesha
"""

"""
You work for a consumer finance company which specializes in lending
loans to urban customers. When the company receives a loan application,
the company has to make a decision for loan approval based on the
applicant’s profile. Two types of risks are associated with the bank’s decision:
- If the applicant is likely to repay the loan,
  then not approving the loan results in a loss of business to the company
- If the applicant is not likely to repay the loan,
  i.e., he/she is likely to default, then approving
  the loan may lead to a financial loss for the company

The data given below contains the information about past loan applicants and whether the loan was approved.

In this case study, you will use EDA to understand how consumer attributes and their
loan approval status are associated.

When a person applies for a loan, there are two types of decisions that could be made:

1. Loan accepted: If the company approves the loan, there are 3 possible scenarios:
   ○ Fully paid: Applicant has fully paid the loan (the principal and the interest)
   ○ Current: Applicant is in the process of paying the installments
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

loan = pd.read_csv("c:/Data-Science/9-Classification/SVM/Loan.csv")

loan.dtypes
loan.head(10)
'''
 id  member_id  ...  total_bc_limit  total_il_high_credit_limit
0  1077501    1296599  ...             NaN                         NaN
1  1077430    1314167  ...             NaN                         NaN
2  1077175    1313524  ...             NaN                         NaN
3  1076863    1277178  ...             NaN                         NaN
4  1075358    1311748  ...             NaN                         NaN
5  1075269    1311441  ...             NaN                         NaN
6  1069639    1304742  ...             NaN                         NaN
7  1072053    1288686  ...             NaN                         NaN
8  1071795    1306957  ...             NaN                         NaN
9  1071570    1306721  ...             NaN                         NaN

'''
loan.columns

# Some of the important columns in the dataset are: loan_amount, term, interest rate
# The target variable, which we want to compare across the independent variables, is

loan.isnull().sum()
'''
id                                0
member_id                         0
loan_amnt                         0
funded_amnt                       0
funded_amnt_inv                   0
 
tax_liens                        39
tot_hi_cred_lim               39717
total_bal_ex_mort             39717
total_bc_limit                39717
total_il_high_credit_limit    39717
Length: 111, dtype: int64

'''
# There are several columns having missing values at higher level
# removing the columns having more than 90% missing values
missing_columns = loan.columns[100*(loan.isnull().sum()/len(loan.index)) > 90]

'''
Step-by-Step Explantion
loan.isnull().sum()
 Counts the number of missing (NaN) values per column.
 Example:
 loan_amount     20
 term             0
 int_rate        15
 ...
len(loan.index)
 Gives the total number of rows in the dataset.
 Example: if there are 10,000 rows.
loan.isnull().sum() / len(loan.index)
 Gives the fraction of missing values per column.
Example: if column loan_amount has 2000 missing out of 10,000 rows
→ 2000 / 10000 = 0.2
100 * (loan.isnull().sum() / len(loan.index))
 Converts the fraction into a percentage of missing values per column.
 Example: 0.2 × 100 = 20%
> 90
Filters only those columns where more than 90% values are missing.
loan.columns[...]
 Returns the names of such columns.
'''
print(missing_columns)
loan = loan.drop(missing_columns, axis=1)
print(loan.shape)
#(39717, 55)

# Summarise number of missing values again
100 * (loan.isnull().sum() / len(loan.index))

# There are desc and mths_since_last_delinqu columns having missing values 32.58%
# 64.66 % let us drop these two columns
# Dropping the two columns
loan = loan.drop(['desc', 'mths_since_last_delinq'], axis=1)
print(loan.shape)
#(39717, 53)

# Summarise number of missing values again
100 * (loan.isnull().sum() / len(loan.index))

# Now let us impute these missing values
# emp_title, emp_length, title, revol_util
loan.dtypes

# grade, sub_grade, emp_title, emp_length, home_ownership, verification_status
# issue_d, loan_status, pymnt_plan, url, purpose, title, zip_code, addr_state,
# earliest_cr_line, revol_util, initial_list_status, last_pymnt_d, 
# last_credit_pull_d, application_type are categorical

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

loan.grade = labelencoder.fit_transform(loan.grade)
loan.sub_grade = labelencoder.fit_transform(loan.sub_grade)
loan.emp_title = labelencoder.fit_transform(loan.emp_title)
loan.emp_length = labelencoder.fit_transform(loan.emp_length)
loan.home_ownership = labelencoder.fit_transform(loan.home_ownership)
loan.verification_status = labelencoder.fit_transform(loan.verification_status)
loan.issue_d = labelencoder.fit_transform(loan.issue_d)
loan.loan_status = labelencoder.fit_transform(loan.loan_status)
loan.pymnt_plan = labelencoder.fit_transform(loan.pymnt_plan)
loan.url = labelencoder.fit_transform(loan.url)
loan.purpose = labelencoder.fit_transform(loan.purpose)
loan.title = labelencoder.fit_transform(loan.title)
loan.zip_code = labelencoder.fit_transform(loan.zip_code)
loan.addr_state = labelencoder.fit_transform(loan.addr_state)
loan.earliest_cr_line = labelencoder.fit_transform(loan.earliest_cr_line)
loan.revol_util = labelencoder.fit_transform(loan.revol_util)
loan.initial_list_status = labelencoder.fit_transform(loan.initial_list_status)
loan.last_pymnt_d = labelencoder.fit_transform(loan.last_pymnt_d)
loan.last_credit_pull_d = labelencoder.fit_transform(loan.last_credit_pull_d)
loan.term = labelencoder.fit_transform(loan.term)
loan.int_rate = labelencoder.fit_transform(loan.int_rate)
loan.application_type = labelencoder.fit_transform(loan.application_type)

loan.dtypes

# ###########3
# summarise number of missing values again
100 * (loan.isnull().sum() / len(loan.index))

import numpy as np
from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

loan['emp_title'] = pd.DataFrame(mean_imputer.fit_transform(loan[['emp_title']]))
loan['collections_12_mths_ex_med'] = pd.DataFrame(mean_imputer.fit_transform(loan[['collections_12_mths_ex_med']]))
loan['chargeoff_within_12_mths'] = pd.DataFrame(mean_imputer.fit_transform(loan[['chargeoff_within_12_mths']]))
loan['pub_rec_bankruptcies'] = pd.DataFrame(mean_imputer.fit_transform(loan[['pub_rec_bankruptcies']]))
loan['tax_liens'] = pd.DataFrame(mean_imputer.fit_transform(loan[['tax_liens']]))

loan.isnull().sum()
##################################################
# In the given data set there are Customer behaviour variables
# (those which are generated after the loan is approved such as delinquent 2 years,
# revolving balance, next payment date etc..).
# They are not really required for the sanction of the loan
# we need variables like 1. those which are related to the applicant
# (demographic variables such as age, occupation, employment details etc.),
# 2. loan characteristics (amount of loan, interest rate, purpose of
# loan etc.)
behaviour_var = [
    "delinq_2yrs",
    "earliest_cr_line",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "revol_util",
    "total_acc",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "last_credit_pull_d",
    "application_type"]
behaviour_var
# let's now remove the behaviour variables from analysis
loan = loan.drop(behaviour_var, axis=1)
loan.dtypes
# Typically, variables such as acc_now_delinquent, chargeoff within 12 months etc. (w
# also, we will not be able to use the variables zip code, address, state etc.
# the variable 'title' is derived from the variable 'purpose'
# thus let get rid of all these variables as well
loan = loan.drop(['title', 'url', 'zip_code', 'addr_state'], axis=1)
loan.shape
#(39717, 28)
loan.dtypes
# our target variable is loan_status, let us shift to 0 th position
loan=loan.iloc[:,[16,0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27]]
# There are several columns having different scale, let us go for normalization
#loan.loc[loan.loan_status>1,"loan_status"]=1
loan['loan_status'].value_counts()
# Fully Paid          32950
# Charged off         5627
# Current             1140
#Fully paid comprises most of the loans.The ones marked 'Current' are neither fully
# let us drop current account because they are of no use
# filtering only fully paid or charged-off
loan = loan[loan['loan_status'] != 1]

loan.loc[loan.loan_status==2, "loan_status"] = 1

# summarising the values
loan['loan_status'].value_counts()
# loan_status
# 1    32950
# 0     5626

############################################################
# loan_status is highly correlated with interest rate
# let us split the data
train, test = train_test_split(loan, test_size=0.2)

# multinomial option is only supported the 'lbfgs','newton rapson -cg' solver
model = LogisticRegression(multi_class="multinomial", solver="newton-cg").fit(train.iloc[:,1:], train.iloc[:,0])
'''
lbfgs = Limited-memory BFGS -> a smart algorithm that finds the best coefficients faster than plain gradient descent.
It is memory-efficient and works well when we have many features.
It is recommended for multinomial Logistic regression.
Other solvers:
newton-cg – also works for multinomial regression.
saga, sag – good for very large datasets.
'''
test_pred = model.predict(test.iloc[:,1:])
#####
accuracy_score(test.iloc[:,0], test_pred)  # 0.8547174701918092
#0.8523846552617936
# Train predict
train_pred = model.predict(train.iloc[:,1:])
accuracy_score(train.iloc[:,0], train_pred)  # 0.8544
#0.8548653640517158

# 6. Write about the benefits/impact of the solution - in what way does the business
# Analysing various factors on loan default, one can easily decide whether to sanction
loan.dtypes
