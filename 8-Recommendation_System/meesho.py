# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 16:19:48 2025

@author: Ashlesha
"""

from bs4 import BeautifulSoup as bs           
import requests    
link='https://www.meesho.com/redmi-a3-6gb-128gb-lake-blue/p/795bx4'                          
page=requests.get(link)                      
page                                           
page.content                                   
soup=bs(page.content,'html.parser')          
print(soup.prettify())                       
title=soup.find_all('p',class_="sc-eDvSVe dugLmN")      
title                                         
review_title=[]                               
for i in range(0,len(title)):                 
    review_title.append(title[i].get_text())  
review_title                                  
len(review_title)   
#####we got 10 review titles
#####Now let us scrap rating
rating=soup.find_all('div',class_='sc-eDvSVe gsatlV')  
rating
rate=[]                                     
for i in range(0,len(rating)):                
    rate.append(rating[i].get_text())        
rate
len(rate)                                     

#################################
#Now let us scarp the review body
review=soup.find_all('div',class_='sc-eDvSVe gsatlV')   
review
review_body=[]                                
for i in range(0,len(review)):                
    review_body.append(review[i].get_text())  
review_body
len(review_body)                              

####we got 10 review_body
###Now we have to save the data in .csv file
import pandas as pd                          
df=pd.DataFrame()                             
df['Review Title']=review_title               
df['Rate']=rate                              
df['Review']=review_body                
df                                           

##################################
##To create .csv file
df.to_csv("c:/Data-Science/8-RecommendationSystem/meesho_reviews.csv")  
##################################

#sentiment analysis
import pandas as pd                          
from textblob import TextBlob                 
sent="This is very excellent garden"          
pol=TextBlob(sent).sentiment.polarity         
pol                                           
df=pd.read_csv("c:/Data-Science/8-RecommendationSystem/meesho_reviews.csv")  
df.head()
df['polarity']=df['Review'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)  
df['polarity']                               
