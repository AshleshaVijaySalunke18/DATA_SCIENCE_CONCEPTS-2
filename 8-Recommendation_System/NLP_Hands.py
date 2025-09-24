# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 18:31:13 2025

@author: Ashlesha
"""

# Test cleaning and Tokenization

import re

sentences = 'Sharat tweeted, "Witnessing 70th Republic Day of India from Rajpath, \
New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! @india_official \
@indian_army #India #70thRepublic_Day. For more photos ping me sharat@photoking.com :)'

re.sub(r'([^\s\w]|_)+','',sentences).split()

#Extracting n-grams
#n-gram can be  extracted from 3 different techniques:
#listed below are:
#1. Custom defined function
#2 NLTK
#3. TextBlob   

#Extracting  n-grams using customed defined function
import re
def n_grams_extractor(input_str,n):
    tokens=re.sub(r'([^\s\w]|_)+','',input_str).split()
    for i in range(len(tokens) - n + 1):
       print(tokens[i:i+n])  
n_grams_extractor('The  cute little boy is  playing with the kitten.',2)
n_grams_extractor('The  cute little boy is  playing with the kitten.',3)


#Extracting n-grams with nltk
from nltk import ngrams
# Bigrams (n=2)
print(list(ngrams('The  cute little boy is  playing with the kitten.'.split(), 2)))
# Trigrams (n=3)
print(list(ngrams('The  cute little boy is  playing with the kitten.'.split(), 3)))

#Extracting n-grams using  TextBlob
#TextBlob is a python  library for proessing  testual data

#pip install textblob            
#python   -m textblob.download_corpora
from textblob import TextBlob
bolb=TextBlob("The  cute little boy is  playing with the kitten.")     
bolb.ngrams(n=2)
bolb.ngrams(n=3)

#Tokenizing texts with different packing : keras,Textblob
sentences = 'Sharat tweeted, "Witnessing 70th Republic Day of India from Rajpath, \
New Delhi. Mesmerizing performance by Indian Army! Awesome airshow! @india_official \
@indian_army #India #70thRepublic_Day. For more photos ping me sharat@photoking.com :)'

#pip install tensorflow
#pip install keras

# Tokenization with keras
from tensorflow.keras.preprocessing.text import text_to_word_sequence

## Example sentence
sentences = "Keras is a deep Learning API written in Python."


tokens = text_to_word_sequence(sentences)
print(tokens)

from textblob import TextBlob
blob = TextBlob(sentences)
blob.words

#1 Tweet rokenizer
#2. MWE Tokenizer (Multi-Word Expression Tokenizer)
#3. RegexpTokenizer#
#4. WhitespaceTokenizer

#1 Tweet rokenizer
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()
tweet_tokenizer.tokenize(sentences)

#2. MWE Tokenizer (Multi-Word Expression Tokenizer)
from nltk.tokenize import MWETokenizer
# Initialize with one MWE
mwe_tokenizer = MWETokenizer([('Republic', 'Day')])
# Add another MWE
mwe_tokenizer.add_mwe(('Indian', 'Army'))
# Tokenize after splitting the sentence into words
mwe_tokenizer.tokenize(sentences.split())

#3. RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+/\$[\d\.]+|\S+')
tokenizer.tokenize(sentences)

#4. WhitespaceTokenizer
from nltk.tokenize import WhitespaceTokenizer
tokenizer = WhitespaceTokenizer()
tokenizer.tokenize(sentences)

#5. WordPunctTokenizer
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
tokenizer.tokenize(sentences)

#Stemming 
#regexp stemming
sentence1="I love  playing cricket. cricket players practice hard."
from nltk.stem import RegexpStemmer
regex_stemmer = RegexpStemmer('ing$')

''.join([regex_stemmer.stem(wd)for wd in sentence1.split()])

#Porter stemmer
sentence2="Before eatting, it  would be nice to  sanitize your hands"
from nltk.stem import PorterStemmer
ps_stemmer = PorterStemmer()
''.join([ps_stemmer(wd)for wd in sentence2.split()])

#Lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# Download the required NLTK resource
nltk.download('wordnet')
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Example sentence
sentence3 = "The codes executed today are far better than what we execute generally."

' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sentence3)])

# Singularize & Pluralize words
from textblob import TextBlob
# Create a TextBlob object with a sentence
sentence4 = TextBlob('She sells seashells on the seashore')

# Display the tokenized words
sentence4.words

# Singularize the third word ('seashells')
sentence4.words[2].singularize()

# Pluralize the sixth word ('seashore')
sentence4.words[5].pluralize()

# Language Translation
# From Spanish to English
# pip install googletrans==4.0.0-rc1
from googletrans import Translator
translator = Translator()
translation = translator.translate("muy bien", src='es', dest='en')
translation = translator.translate("good night", src='en', dest='hi')
print("Translated text:", translation.text)


# Custom Stop words removal
from nltk import word_tokenize
sentence9 = "She sells seashells on the seashore"
custom_stop_word_list = ['she', 'on', 'the', 'am', 'is', 'not']
' '.join([word for word in word_tokenize(sentence9) if word.lower() not in custom_stop_word_list])


#26-6-25

#Extracting general  features from raw texts

#no of word
#Detect presence of wh words
#polarity
#subjectivityS
#language identification

import pandas as pd
df = pd.DataFrame({'text': ["The quick brown fox jumps over the lazy dog."]})
df.columns=['text']
df
# Import TextBlob
from textblob import TextBlob

# 1. Number of words
df['number_of_words'] = df['text'].apply(lambda x: len(TextBlob(x).words))
df['number_of_words']

# 2. Detect presence of wh-words
wh_words = set(['why', 'who', 'which', 'what', 'where', 'when', 'how'])
df['is_wh_words_present'] = df['text'].apply(
    lambda x: True if len(set(TextBlob(str(x)).words).intersection(wh_words)) > 0 else False
)
df['is_wh_words_present']


# 3. Sentiment Polarity
df['polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['polarity']

# 4. Subjectivity
'''
Subjectivity is a measure of how much a 
piece of text expresses personal opinions, 
feelings, or beliefs – as opposed to objective facts.
Scale:
0.0 = Completely objective (facts, data, unbiased statements)
1.0 = Completely subjective (opinions, emotions, personal views)
'''

df['subjectivity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
df['subjectivity']
#pip install langdetect
# Language of the sentence
from langdetect import detect
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'text': [
        "Hello, how are you?",
        "Hola, ¿cómo estás?",
        "Bonjour tout le monde"
    ]
})

# 5 . Detect language
df['language']=df['text'].apply(lambda x:detect(str(x)))
print(df[['text','language']])

'''
|code|language|
|'en|english|
'''

#27-6-25

# Feature Engineering (Text Similarity)

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = WordNetLemmatizer()

pair1 = ["Do you have Covid-19", "Your body temperature will tell you"]
pair2 = ["I travelled to Malaysia.", "Where did you travel?"]
pair3 = ["He is a programmer", "Is he not a programmer?"]

# ------------------- Jaccard Similarity Function -------------------
'''
Jaccard Similarity Formula:
Jaccard Similarity = Intersection of sets / Union of sets

It returns a value between:
0: no similarity
1: exact match (identical word sets)
'''


def extract_text_similarity_jaccard(text1, text2):
    # Normalize and lemmatize text1
    words_text1 = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text1)]
    
    '''
    word_tokenize(text1): Splits text1 into individual words.
    .lower(): Converts each word to lowercase to avoid case mismatches.
    lemmatizer.lemmatize(...): Reduces each word to its base form (e.g., running → run)
    The result: a normalized list of words from text1
    '''

    # Normalize and lemmatize text2
    words_text2 = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text2)]
    intersection = len(set(words_text1).intersection(set(words_text2)))
    union = len(set(words_text1).union(set(words_text2)))
    # Counts the total unique words present in either of the texts.
    jaccard_sim = intersection / union if union > 0 else 0
    # Calculates the ratio of common to total words.
    # If union is 0 (to avoid division by zero), it returns 0.
    return jaccard_sim
'''
extract_text_similarity_jaccard("He is a developer", "Is he not a developer")
After preprocessing:
Set1 = {"he", "is", "a", "developer"}
Set2 = {"is", "he", "not", "a", "developer"}
Intersection = {"he", "is", "a", "developer"} → 4
Union = {"he", "is", "a", "not", "developer"} → 5
So, Jaccard = 4/5 = 0.8
'''
# Display Jaccard similarity
print("Jaccard Similarity:")
print("Pair 1:", extract_text_similarity_jaccard(pair1[0], pair1[1]))
print("Pair 2:", extract_text_similarity_jaccard(pair2[0], pair2[1]))
print("Pair 3:", extract_text_similarity_jaccard(pair3[0], pair3[1]))

# ------------------ TF-IDF Cosine Similarity ------------------#

# Correct corpus creation
corpus = [pair1[0], pair1[1], pair2[0], pair2[1], pair3[0], pair3[1]]

# Vectorization
tfidf_model = TfidfVectorizer()
tfidf_results = tfidf_model.fit_transform(corpus)

# Cosine similarities
print("\nCosine Similarity:")
print("Pair 1:", cosine_similarity(tfidf_results[0], tfidf_results[1])[0][0])
print("Pair 2:", cosine_similarity(tfidf_results[2], tfidf_results[3])[0][0])
print("Pair 3:", cosine_similarity(tfidf_results[4], tfidf_results[5])[0][0])
