# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 09:38:14 2025

@author: Ashlesha
"""

#7-3-25
#validation
def validate_experience(exp):
    """Validate that experience is a non-negative integer."""
    if not isinstance(exp, int) or exp < 0:
        raise ValueError("Experience must be a non-negative integer.")
    return exp

def validate_role(role):
    """Validate job role."""
    valid_roles = {"Intern", "Junior", "Mid-level", "Senior", "Manager"}
    role = role.capitalize()  # Normalize input to match valid roles

    if role not in valid_roles:
        raise ValueError(f"Invalid role! Choose from {', '.join(valid_roles)}.")
    
    return role


    