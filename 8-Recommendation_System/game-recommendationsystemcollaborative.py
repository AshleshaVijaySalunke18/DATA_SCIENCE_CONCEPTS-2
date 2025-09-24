# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:50:30 2025

@author: Ashlesha
"""

# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the CSV file
file_path = "C:/Data-Science/8-RecommendationSystem/game.csv"
data = pd.read_csv(file_path)

#userID-ID of the user
#game- name of the game
#rating -rating given by the user to the game

#step1 create a user -item matrix (rows:user,column:game,value)
user_item_matrix=data.pivot_table(index='userId',columns='game',values='rating')

'''
pivot_table: This function reshapes the DataFrame into a matrix where:

    Each row represents a user (identified by userId).
    Each column represents a game (identified by game).
    The values in the matrix represent the ratings that users gave to the games.
'''

# Step 2: Fill NaN values with 0 (assuming no rating means the game has not been rated)
user_item_matrix_filled = user_item_matrix.fillna(0)

'''
This line replaces any missing values (NaNs)
in the user-item matrix with 0,
indicating that the user did not rate that particular game.
'''

# Step 3: Compute the cosine similarity between users based on raw ratings
user_similarity = cosine_similarity(user_item_matrix_filled)

# Convert similarity matrix to a DataFrame for easy reference
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Step 4: Function to get game recommendations for specific user based on similarity
# This function takes a user ID and gives top N recommended games
def get_collaborative_recommendations_for_user(user_id, num_recommendations=5):
    # Get the similarity score of this user with all other users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    
    #get the most similar user (excluding the user themselves)
    similar_users=similar_users.drop(user_id)
    
    #select the top N simplar user  to limit noise (e.g, top 50 user)
    top_similar_user=similar_users.head(50)
    #this select the top 50 most silar user  to limit
    #nosie in the recommendation
    #get rating of those similar  user, weighted by their
    #similarity score
    weighted_ratings=np.dot(top_similar_user.values,user_item_matrix_filled.loc[top_similar_user.index])
    #multiply rating of similar users with their similarity score
    
    #normalize it to  prevent bias toward users with high values.
    #np.dot:this  computes the dot product between the 
    #similarity score of the  top  similar user and their corresponding rating
    #normalized by the sum of  similarities
    sum_of_similarities=top_similar_user.sum()
    
    if sum_of_similarities>0:
        weighted_ratings/=sum_of_similarities
        
        #the weighted rating  are normalized by dividing by the
        #sum of similarities to avoid biasing toward  users with highter rating
        
        #recommend game that the uesr has't rated yet
    user_ratings=user_item_matrix_filled.loc[user_id]
    unrated_games=user_ratings[user_ratings==0]
    
#this  identifies game  that the target user has not rated
     # Get the weighted scores for unrated games
    game_recommendations = pd.Series(
        weighted_ratings,
        index=user_item_matrix_filled.columns
    ).loc[unrated_games.index]
    
    # This creates a pandas Series from the weighted ratings
    # and filters it to include only the unrated games.
    # Finally, it sorts the recommendations in descending order
    # and returns the top specified number of recommendations.
    
    # Return the top 'num_recommendations' game recommendations
    return game_recommendations.sort_values(ascending=False).head(num_recommendations)

#example usege:get recommendation for a user with id 3
recommend_games=get_collaborative_recommendations_for_user(user_id=3)

#print the recommend game
print("Recommended game for user 3:")
print(recommend_games)

     
