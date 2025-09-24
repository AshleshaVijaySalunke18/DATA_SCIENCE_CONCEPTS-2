# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 16:40:50 2025

@author: Ashlesha
"""

#------------------ pip install bs4------------------#

from bs4 import BeautifulSoup
soup = BeautifulSoup(open("c:/Data-Science/8-RecommendationSystem/sample_doc.html"), 'html.parser')
print(soup)
# it is going to show all the html contents extracted
soup.text
# It will show only text
soup.contents
# it is going to show all the html contents extracted
soup.find('address')
soup.find_all('address')
soup.find_all('q')
soup.find_all('b')
table = soup.find('table')
table
for row in table.find_all('tr'):
    columns = row.find_all('td')
    print(columns)

# It will show all the rows except first row
# Now we want to display M.Tech which is located in third row and second column
# I need to give [3][2]
    table.find_all('tr')[3].find_all('td')[2]
