# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:34:47 2025

@author: Ashlesha
"""


from bs4 import BeautifulSoup as bs           
import requests                              
link='https://www.flipkart.com/samsung-1-5-ton-5-star-split-inverter-ac-white/p/itm1a2f33df93c02?pid=ACNGADBFVHZE4ZFV&lid=LSTACNGADBFVHZE4ZFVRB2YHV&marketplace=FLIPKART&fm=neo%2Fmerchandising&iid=M_07c6b85c-751d-456e-a7ce-3f86576fe909_4_7JLC3S5TXLAG_MC.ACNGADBFVHZE4ZFV&ppt=hp&ppn=homepage&otracker=clp_pmu_v2_Air%2BConditioners_3_4.productCard.PMU_V2_SAMSUNG%2B1.5%2BTon%2B5%2BStar%2BSplit%2BInverter%2BAC%2B%2B-%2BWhite_acnewclp-store_ACNGADBFVHZE4ZFV_neo%2Fmerchandising_2&otracker1=clp_pmu_v2_PINNED_neo%2Fmerchandising_Air%2BConditioners_LIST_productCard_cc_3_NA_view-all&cid=ACNGADBFVHZE4ZFV'  
page=requests.get(link)                      
page                                           
page.content                                   
soup=bs(page.content,'html.parser')          
print(soup.prettify())                       
title=soup.find_all('p',class_="z9E0IG")      
title                                         
review_title=[]                               
for i in range(0,len(title)):                 
    review_title.append(title[i].get_text())  
review_title                                  
len(review_title)                             

#####we got 10 review titles
#####Now let us scrap rating
rating=soup.find_all('div',class_='XQDdHH Ga3i8K')  
rating
rate=[]                                     
for i in range(0,len(rating)):                
    rate.append(rating[i].get_text())        
rate
len(rate)                                     

#################################
#Now let us scarp the review body
review=soup.find_all('div',class_='ZmyHeo')   
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
df.to_csv("c:/Data-Science/8-RecommendationSystem/flipkart_reviews.csv")  
##################################

#sentiment analysis
import pandas as pd                          
from textblob import TextBlob                 
sent="This is very excellent garden"          
pol=TextBlob(sent).sentiment.polarity         
pol                                           
df=pd.read_csv("c:/Data-Science/8-RecommendationSystem/flipkart_reviews.csv")  
df.head()
df['polarity']=df['Review'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)  
df['polarity']                               
