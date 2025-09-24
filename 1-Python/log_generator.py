# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 15:16:13 2025

@author: Ashlesha
"""

import time
import random
LOG_FILE="c:/Data-Science/1-python/server.log"
LOG_LEVELS=["INFO","WARNING","ERROR","CRITICAL"]
MESSAGES=[
    "user  logged in:user123",
    "high memory usage detected",
    "database connection failed:timeout",
    "file uploaded:report.pdf",
    "server crash detected! restarting..",
    "user logged out:user456",
    ]
def generate_logs():
    '''Continuously writes logs to a file every 2 seconds'''
    with open(LOG_FILE, 'a') as file:
        while True:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_level = random.choice(LOG_LEVELS)
            message = random.choice(MESSAGES)
            log_entry = f"{timestamp} [{log_level}] {message}\n"
            file.write(log_entry)
            file.flush()
            print("Generated")



