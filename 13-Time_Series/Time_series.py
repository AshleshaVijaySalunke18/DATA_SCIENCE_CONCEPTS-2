# -*- coding: utf-8 -*-
"""
3-9-2025

@author: Ashlesha
"""

# Working with Dates, Times, NumPy datetime64, and Pandas
# --------------------------------------------------------

from datetime import datetime
import numpy as np
import pandas as pd

# --------------------------------------------------------
# 1. Python datetime Module
# --------------------------------------------------------

# Define components of a date and time
my_year = 2025
my_month = 1
my_day = 2
my_hour = 13
my_minute = 30
my_second = 15

# Create a datetime object with only year, month, and day
# Defaults the time part to 00:00:00 (midnight)
my_date = datetime(my_year, my_month, my_day)
print("Date only:", my_date)

# Create a datetime object with full date and time
my_date_time = datetime(my_year, my_month, my_day,
                        my_hour, my_minute, my_second)
print("Date and time:", my_date_time)

# Extract just the day from a datetime object
print("Day of the month:", my_date.day)

# Extract just the hour from a datetime object
print("Hour of the day:", my_date_time.hour)

# --------------------------------------------------------
# 2. NumPy datetime64
# --------------------------------------------------------

# Create a NumPy array of dates (default precision is days)
dates_day_precision = np.array(
    ['2016-03-15', '2017-05-24', '2018-08-09'], dtype='datetime64')
print("\nDates (day precision):", dates_day_precision)

# Specify precision as hours
dates_hour_precision = np.array(
    ['2016-03-15', '2017-05-24', '2018-08-09'], dtype='datetime64[h]')
print("Dates (hour precision):", dates_hour_precision)

# Specify precision as years
dates_year_precision = np.array(
    ['2016-03-15', '2017-05-24', '2018-08-09'], dtype='datetime64[Y]')
print("Dates (year precision):", dates_year_precision)

# Create a range of dates from June 1, 2018 to June 22, 2018
# spaced one week apart
weekly_dates = np.arange('2018-06-01', '2018-06-23', 7, dtype='datetime64[D]')
print("Weekly dates:", weekly_dates)

# Create a range of yearly dates from 1968 to 1975
yearly_dates = np.arange('1968', '1976', dtype='datetime64[Y]')
print("Yearly dates:", yearly_dates)

# --------------------------------------------------------
# 3. Pandas Datetime
# --------------------------------------------------------

import pandas as pd

# A list of dates written in different formats
dates = ['Jan 01, 2018', '1/2/18', '03-Jan-2018', None]

# Convert each date safely.
# If a value cannot be understood (like None), it becomes NaT (Not a Time)
parsed = [pd.to_datetime(d, errors='coerce', dayfirst=True) for d in dates]

'''
for d in dates
This is a loop inside a list comprehension.
It takes each item from the list dates one by one.
Example: first d = 'Jan 01, 2018', then '1/2/18', then '03-Jan-2018', then None
pd.to_datetime(d, errors='coerce', dayfirst=True)
Converts the value d into a pandas datetime object.
errors='coerce' ➝ If the value cannot be converted (like None or a wrong format), it becomes NaT.
dayfirst=True ➝ Interprets ambiguous dates like '1/2/18' as 1-Feb-2018 (day/month/year)
[ ... ] ➝ Square brackets denote list comprehension output
 This creates a list of all the converted dates.
So after the loop finishes, you get a list of datetime objects.
'''
# Put them into a DatetimeIndex (like a list of pandas dates)
idx = pd.DatetimeIndex(parsed)

print("Original List:", dates)
print("Parsed dates :", idx)

# --------------------------------------------------------
# 4. Using Pandas DatetimeIndex with DataFrame
