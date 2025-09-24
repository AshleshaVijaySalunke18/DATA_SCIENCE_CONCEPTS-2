import pandas as pd
game1 = pd.read_csv("C:/Data-Science/8-RecommendationSystem/game.csv", encoding='utf8')
game1.shape
#you will get (5000, 3) matrix
game1.columns
game1. head()

#here we are considering only genre
from sklearn.feature_extraction.text import TfidfVectorizer
#this is term frequency inverse document
#each row is treated as document
tfidf=TfidfVectorizer(stop_words='english')
#checking for  nan values
game1['rating'].isna().sum()
tfidf_matrix=tfidf.fit_transform(game1.game)
tfidf_matrix.shape
#measure the  simplarity using cosine  similarity
from sklearn.metrics.pairwise import linear_kernel
#creating cosine similarity matrix which will create of similar
cos_sim_matrix=linear_kernel(tfidf_matrix, tfidf_matrix)
#we will create series of game1
game1_index = pd.Series(game1.index, index=game1['userId'])
game1.head()
#checking the  same for  randam  game picked up
game1_id=game1_index[269]
game1_id
#now let us create user defined function
def get_recommendation(userId,topN):
    #getting game  index and its user id
    game1_id=game1_index[userId]
    #getting pairwise similarity score
    cosine_scores=list(enumerate(cos_sim_matrix[game1_id]))
    cosine_scores=sorted(cosine_scores,key=lambda x:x[1],reverse=True)
    cosine_scores_N=cosine_scores[0: topN + 1]
    #getting  game  index
    game1_idx=[i[0] for i in cosine_scores_N]
    game1_scores=[i[1]for i in cosine_scores_N]
    games_similar=pd.DataFrame(columns=["game","rating"])
    games_similar['game']=game1.loc[game1_idx,'game']
    games_similar['rating']=game1_scores
    games_similar.reset_index(inplace=True)
    print(games_similar)
    
 #let us use this  function which will give topN gamelist
get_recommendation(285,topN=10)
game1_index[285]   























