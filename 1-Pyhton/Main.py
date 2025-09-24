# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 08:28:31 2025

@author: Ashlesha
"""
#10-3-25
#IMPORTANT
#main
from salary import calculate_salary
from validation import validate_experience, validate_role
from display import display_salary

def main():
    """Main function to get input and calculate salary."""
    try:
        name = input("Enter employee name: ")
        experience = int(input("Enter years of experience: "))
        role = input("Enter job (Intern, Junior, Mid-level, Senior, Manager): ")

        # Validate inputs
        experience = validate_experience(experience)
        role = validate_role(role)

        # Calculate salary
        salary = calculate_salary(experience, role)

        # Display result
        display_salary(name, experience, role, salary)

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
