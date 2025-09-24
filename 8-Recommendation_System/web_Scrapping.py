# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:06:08 2025

@author: Ashlesha
"""
# 2 code 
#1NEWS
#2LOCATION

#------------------------------------------------
#1NEWS
from bs4 import BeautifulSoup as bs
import requests

link = "https://www.lokmat.com/maharashtra/good-news-for-maharashtra"
page = requests.get(link)
page
# <Response [200]> it means connection is successfully established
page.content  # you will get all HTML source code but very crowded text
# Let us apply HTML parser
soup = bs(page.content, 'html.parser')
soup
# Now the text is clean but not up to the expectations
# Now let us apply prettify method
print(soup.prettify())
# The text is neat and clean
list(soup.children)
# Finding all contents using tag
soup.find_all('p')
# Suppose you want to extract contents from specific rows
# First row
soup.find_all('p')[1].get_text()
# Contents from second row
soup.find_all('p')[2].get_text()
# Finding text using class
soup.find_all('div', class_='table')

#--------------------------------------------------------

#------2 LOCATION----------

from bs4 import BeautifulSoup as bs         
import requests                              
link = "https://sanjivanicoe.org.in/index.php/contact"  
page = requests.get(link)                   
page                                        
page.content                                
#you will get all html source code but very crowdy text
#let us apply html parser
soup = bs(page.content,'html.parser')       # Parse the HTML using html.parser
soup                                        # Display parsed HTML soup object
#Now the text is clean but not upto the expectations
#Now let us apply prettify method
print(soup.prettify())                      # Neatly formatted (indented) HTML
#The text is neat and clean
list(soup.children)                         # List top-level HTML elements (like html, head, body)
#Finding all contents using tab
soup.find_all('p')                          # Find all <p> tags in the HTML
#suppose you want to extract contents from
#first row
soup.find_all('p')[1].get_text()            # Extract plain text from the second <p> tag
#contents from second row
soup.find_all('p')[2].get_text()            # Extract plain text from the third <p> tag
#finding text using class
soup.find_all('div',class_='table')         # Find all <div> tags with class "table"
