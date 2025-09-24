# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:23:31 2025

@author: Ashlesha
"""

# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
file_path = "C:/Data-Science/8-RecommendationSystem/Entertainment.csv"
data = pd.read_csv(file_path)

# Step 1: Normalize the review scores using MinMaxScaler (0 to 1 scale)
scaler = MinMaxScaler()
data['Normalized_Reviews'] = scaler.fit_transform(data[['Review']])

# Step 2: Compute cosine similarity between titles based on normalized reviews
cosine_sim_reviews = cosine_similarity(data[['Normalized_Reviews']])

# Step 3: Create a function to recommend titles based on similarity
def get_collaborative_recommendation(title, cosine_sim=cosine_sim_reviews):
    if title not in data['Titles'].values:
        return f"Title '{title}' not found in the dataset."

    # Get the index of the title that matches the input
    idx = data[data['Titles'] == title].index[0]

    # Get the pairwise similarity scores for that title
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort titles based on similarity score (descending), skip the first (itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    # Get indices of the top 5 similar titles
    sim_indices = [i[0] for i in sim_scores]

    # Return top 5 recommended titles
    return data['Titles'].iloc[sim_indices]

# Test the recommendation system with a title
example_title = "Toy Story (1995)"
collaborative_recommended_titles = get_collaborative_recommendation(example_title)

  
    
    
    