# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 15:37:41 2025

@author: Ashlesha
"""

import pandas as pd
anime = pd.read_csv("C:/Data-Science/7-PCA-SVD/anime.csv", encoding='utf8')
anime.shape
#you will get 12294*7 matrix
anime.columns
anime.genre
#here we are considering only genre
from sklearn.feature_extraction.text import TfidfVectorizer
#this is term frequency inverse document
#each row is treated as document
tfidf=TfidfVectorizer(stop_words='english')
#it is going to create TfidfVectorizer to seperate all stop words.it is going
#out all words from  the row 
#now let us check is there any null value
anime['genre'].isnull().sum()
#there are 62 null values
#suppose one  movie has got genre drams,romance,.. there may be many empty space
#so let us  impute these empty spaces,general is like simple imputer
anime['genre']=anime['genre'].fillna('general')
#now  let us create tfidf_matrix
tfidf_matrix=tfidf.fit_transform(anime.genre)
tfidf_matrix.shape
#you will get 12294,47
#it has create sparse matrix, it means that we have  47 genre
#on this  created sparse   martrix , item  based recommendation
#if a user has watched gadar,then you can recommend  shershah movie
from sklearn.metrics.pairwise import  linear_kernel
#this  is for  measuring  similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
#each element  of  tfidf_matrix is compared with each element of tfidf_matrix
#output will be  similarity matrixof  size 12294*12294 size
#here in cosine_sim_matrix, there are no movie  name only index are provide
#for that purpose custom function is written
anime_index=pd.Series(anime.index,index=anime['name']).drop_duplicates()
#We are converting anime_index into series format,we want index and corresp
anime_id=anime_index['Assassins (1995)']
anime_id
def get_recommendations(Name,topN):
    #topN=10
    anime_id=anime_index[Name]

    #We want to capture whole row of given movie name,its score and column id
    #For that purpose we are applying cosine_sim_matrix to enumerate function
    #Enumerate function create a object which we need to create in list form
    #we are using enumerate function, what enumerate does,
    #suppose we have given
    #(2,10,15,18),if we apply to enumerate then it will create a list
    #(0,2 , 1,10,  3,15,  4,18)
    cosine_scores=list(enumerate(cosine_sim_matrix[anime_id]))
    #The cosine scores captured,we want to arrange in descending order so that
    #we can recommend top 10 based on highest similarity i.e. score
    #if we will check the cosine score, it comprises of index:cosine score
    #x[0]=index and x[1] is cosine score
    #we want arrange tuples in descending order of the score not index
    # Sorting the cosine_similarity scores based on scores i.e x[1]
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
    # The cosine scores captured, we want to arrange in descending order so that
    # we can recommend top 10 based on highest similarity i.e. score
    # If we will check the cosine score, it comprises of index:cosine score
    # x[0] = index and x[1] is cosine score
    # We want to arrange tuples according to decreasing order of the score not index
    
    # Getting the cosine similarity scores for the given anime
    cosine_scores = list(enumerate(cosine_sim_matrix[anime_id]))
    
    # Sorting the cosine_similarity scores based on scores i.e. x[1]
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of top N most similar movies
    # To capture TopN movies, you need to give topN + 1
    cosine_scores_N = cosine_scores[0: topN + 1]
    
        # getting the movie index
    anime_idx = [i[0] for i in cosine_scores_N]
    
    # getting cosine score
    anime_scores = [i[1] for i in cosine_scores_N]
    
    # We are going to use this information to create a dataframe
    # Create an empty dataframe
    anime_similar_show = pd.DataFrame(columns=['name', 'score'])
    
    # Assign anime_idx to name column
    anime_similar_show['name'] = anime.loc[anime_idx, 'name']
    
    # Assign score to score column
    anime_similar_show['score'] = anime_scores
    
    # While assigning values, it is by default capturing original index of the data
    # We want to reset the index
    anime_similar_show.reset_index(inplace=True)
    
    print(anime_similar_show)
    
# Enter your anime and number of animes to be recommended
get_recommendations('Bad Boys (1995)', topN=10)
    
        


