# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 15:53:31 2025

@author: Ashlesha
"""

# !pip install gensim
# !pip install python-Levenshtein

import gensim
import pandas as pd

df = pd.read_json("C:/Data-Science/8-RecommendationSystem/Cell_Phones_and_Accessories_5.json", lines=True)
df
df.shape

# Simple Preprocessing & Tokenization
review_text = df.reviewText.apply(gensim.utils.simple_preprocess)

'''
    simple_preprocess:
    Converts text to lowercase
    Removes punctuation, special characters, etc.
    Tokenizes the text into words.
    review_text is now a list of words for each review.
    Example:
    df.reviewText.loc[0]
    # "I love this charger! It's ..."
  # Example -
df.reviewText.loc[0]
# "I love this charger! It's very fast and reliable."
review_text.loc[0]
# ['love', 'this', 'charger', 'it', 'very', 'fast', 'and', 'reliable']


'''
review_text
#let us check first word of embedding

# Training the Word2Vec Model
model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4,
)

'''
window=10: Considers 10 words to the left and right of the target word as context.
min_count=2: Ignores words that appear fewer than 2 times in the entire corpus.
workers=4: Uses 4 CPU threads for faster training.
'''

# Build Vocabulary
model.build_vocab(review_text, progress_per=1000)

'''
Creates a vocabulary from the tokenized text.
progress_per=1000: Prints progress for every 1000 documents.
'''

# progress_per: after 1000 words it shows progress
# Train the Word2Vec Model
# it will take time, have patience

model.train(review_text, total_examples=model.corpus_count,epochs=model.epochs)

'''
Trains the Word2Vec neural network on your tokenized corpus.
Learns vector representations (embeddings) for words based on their context.
'''

# Save the Model
model.save("C:/Data-Science/8-RecommendationSystem/word2vec_Cell_Phones_and_Accessories-reviews-shor.model")

# Finding Similar Words and Similarity between words
model.wv.most_similar("bad")
model.wv.similarity(w1="cheap", w2="inexpensive")
model.wv.similarity(w1="great", w2="good")

    
