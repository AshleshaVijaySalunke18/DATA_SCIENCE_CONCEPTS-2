# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 08:17:55 2025

@author: Ashlesha
"""

###########################################
    #10-3-25
###########################################
#display

def display_salary(name,experience,role,salary):
   """ Display the salary in a formated way"""
   print("\n== salary Details===")
   print(f"Employee Name:{name}")
   print(f"ROle:{role}")
   print(f"Experience:{experience}year")
   print(f"Calculated Salary:${salary:,.2f}")
   print("================")
   