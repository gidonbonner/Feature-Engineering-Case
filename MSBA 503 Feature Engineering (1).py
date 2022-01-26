# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:27:18 2021

@author: gidonbonner
"""
# Import all necessary libraries
import numpy as np
import pandas as pd 
from bs4 import BeautifulSoup
from requests_html import HTMLSession
import re
import unicodedata
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#from sklearn_pandas import DataFrameMapper
import matplotlib.pyplot as plt
from io import StringIO

# Create file_path variable to easily reference file path for reading files
file_path = 'C:/Users/bonne/Downloads/'

# Read AllData and FS_Sentiment_Dictionary files 
AllData = pd.read_pickle(file_path + 'AllData')
FS_Sentiment_Dictionary = file_path + 'FS_Sentiment_Dictionary.csv'

# Create Dataframe for Sentiment Dictionary
Sentiment_Dictionary = pd.read_csv(FS_Sentiment_Dictionary)

# Create variables for all the different sentiment words
Positive_Words = Sentiment_Dictionary.loc[(Sentiment_Dictionary.Positive == 1),['Word']]
Negative_Words = Sentiment_Dictionary.loc[(Sentiment_Dictionary.Negative == 1),['Word']]
Uncertainty_Words = Sentiment_Dictionary.loc[(Sentiment_Dictionary.Uncertainty == 1),['Word']]
Litigious_Words = Sentiment_Dictionary.loc[(Sentiment_Dictionary.Litigious == 1),['Word']]
Constraining_Words = Sentiment_Dictionary.loc[(Sentiment_Dictionary.Constraining == 1),['Word']]
Superfluous_Words = Sentiment_Dictionary.loc[(Sentiment_Dictionary.Superfluous == 1),['Word']]
Interesting_Words = Sentiment_Dictionary.loc[(Sentiment_Dictionary.Interesting == 1),['Word']]
Modal_Words = Sentiment_Dictionary.loc[(Sentiment_Dictionary.Modal == 1),['Word']]

# Used a function to determine sentiment scores for the different sentiments
def sentiment_count(text, sentiment_text):
    results = []
    blank_words = 0
    for sentence in text: 
        blank_words = 0
        for word in sentence:
            if word in list(sentiment_text):  
                blank_words = blank_words + 1
                sentiment_score = blank_words / len(sentence)
            elif blank_words == 0:
                sentiment_score = 0
        
        results.append(sentiment_score)
    return results

# Add new columns to AllData dataframe for the different sentiment scores using the function defined above
AllData['Positive_Score'] = sentiment_count(AllData['MDA_List'], Positive_Words.Word)
AllData['Negative_Score'] = sentiment_count(AllData['MDA_List'], Negative_Words.Word)
AllData['Uncertain_Score'] = sentiment_count(AllData['MDA_List'], Uncertainty_Words.Word)
AllData['Litigious_Score'] = sentiment_count(AllData['MDA_List'], Litigious_Words.Word)
AllData['Constraining_Score'] = sentiment_count(AllData['MDA_List'], Constraining_Words.Word)
AllData['Superfluous_Score'] = sentiment_count(AllData['MDA_List'], Superfluous_Words.Word)
AllData['Interesting_Score'] = sentiment_count(AllData['MDA_List'], Interesting_Words.Word)
AllData['Modal_Score'] = sentiment_count(AllData['MDA_List'], Modal_Words.Word)

# Check that all sentiment scores were added to the AllData dataframe
AllData.loc[:,'Positive_Score':'Modal_Score']

# Standardize sentiment score columns by subtracting the mean and dividing by the standard deviation
AllData['Positive_Score'] = (AllData.Positive_Score - AllData.Positive_Score.mean()) / AllData.Positive_Score.std()
AllData['Negative_Score'] = (AllData.Negative_Score - AllData.Negative_Score.mean()) / AllData.Negative_Score.std()
AllData['Uncertain_Score'] = (AllData.Uncertain_Score - AllData.Uncertain_Score.mean()) / AllData.Uncertain_Score.std()
AllData['Litigious_Score'] = (AllData.Litigious_Score - AllData.Litigious_Score.mean()) / AllData.Litigious_Score.std()
AllData['Constraining_Score'] = (AllData.Constraining_Score - AllData.Constraining_Score.mean()) / AllData.Constraining_Score.std()
AllData['Superfluous_Score'] = (AllData.Superfluous_Score - AllData.Superfluous_Score.mean()) / AllData.Superfluous_Score.std()
AllData['Interesting_Score'] = (AllData.Interesting_Score - AllData.Interesting_Score.mean()) / AllData.Interesting_Score.std()
AllData['Modal_Score'] = (AllData.Modal_Score - AllData.Modal_Score.mean()) / AllData.Modal_Score.std()

# Check that all the sentiment score columns were standardized
AllData.loc[:,'Positive_Score':'Modal_Score']

# Create a copy of the AllData dataframe
MDA_Copy = AllData.copy()

# Import nltk library and necessary sub packages for lemmatizing
import nltk
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
wnl = nltk.WordNetLemmatizer()

# Use a function to lemmatize the MDA section
def lemmatization(token_MDA):
    tokens = re.split('\W+', token_MDA)
    token = ' '.join([(wnl.lemmatize(word.lower())).upper() for word in tokens])
    return token

# Add new column to MDA_Copy dataframe for the lemmatized MDA section
MDA_Copy['MDA_Lemmatized'] = MDA_Copy['MDA'].apply(lambda x: lemmatization(x))

# Check the difference between the original MDA section and the lemmatized MDA section
MDA_Copy.loc[:,['MDA','MDA_Lemmatized']]

# Import nltk library and necessary sub packages for stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Use a function to stemm the MDA section
def stemming(MDA_tokens):
    tokens = re.split('\W+', MDA_tokens)
    text = ' '.join([(ps.stem(word)).upper() for word in tokens])
    return text

# Add new column to MDA_Copy dataframe for the stemmed MDA section 
MDA_Copy['MDA_Stemmed'] = MDA_Copy['MDA_Lemmatized'].apply(lambda x: stemming(x))

# Check the difference between the original MDA section and the stemmed MDA section
MDA_Copy.loc[:,['MDA','MDA_Stemmed']]

# Create document term matrix for MDA section
corpus_MDA = MDA_Copy.MDA
v = TfidfVectorizer(max_features = 1000)
document_term_matrix_MDA = v.fit_transform(corpus_MDA)
print(document_term_matrix_MDA.shape)

# Create document term matrix for lemmatized MDA section
corpus_MDA_Lemmatized = MDA_Copy.MDA_Lemmatized
v = TfidfVectorizer(max_features = 1000)
document_term_matrix_MDA_Lemmatized = v.fit_transform(corpus_MDA_Lemmatized)
print(document_term_matrix_MDA_Lemmatized.shape)

# Create document term matrix for stemmed MDA section
corpus_MDA_Stemmed = MDA_Copy.MDA_Stemmed
v = TfidfVectorizer(max_features = 1000)
document_term_matrix_MDA_Stemmed = v.fit_transform(corpus_MDA_Stemmed)
print(document_term_matrix_MDA_Stemmed.shape)

# Truncate MDA document term matrix into 10 components
truncatedSVD = TruncatedSVD(10)
document_term_matrix_MDA_truncated = truncatedSVD.fit_transform(document_term_matrix_MDA)
document_term_matrix_MDA_truncated[:10]

# Truncate lemmatized MDA document term matrix into 10 components
document_term_matrix_MDA_Lemmatized_truncated = truncatedSVD.fit_transform(document_term_matrix_MDA_Lemmatized)
document_term_matrix_MDA_Lemmatized_truncated[:10]

# Truncate stemmed MDA document term matrix into 10 components
document_term_matrix_MDA_Stemmed_truncated = truncatedSVD.fit_transform(document_term_matrix_MDA_Stemmed)
document_term_matrix_MDA_Stemmed_truncated[:10]

# Create new dataframe for truncated MDA document term matrix and rename columns to MDA_variables
MDA_truncated = pd.DataFrame(document_term_matrix_MDA_truncated)
MDA_truncated.rename(columns={0:'MDA_var1', 1:'MDA_var2', 2:'MDA_var3', 3:'MDA_var4', 4:'MDA_var5', 5:'MDA_var6', 6:'MDA_var7', 7:'MDA_var8', 8:'MDA_var9', 9:'MDA_var10'}, inplace=True)

# Create new dataframe for truncated lemmatized MDA document term matrix and rename columns to MDA_Lemmatized variables
MDA_Lemmatized_truncated = pd.DataFrame(document_term_matrix_MDA_Lemmatized_truncated)
MDA_Lemmatized_truncated.rename(columns={0:'MDA_Lemmatized_var1', 1:'MDA_Lemmatized_var2', 2:'MDA_Lemmatized_var3', 3:'MDA_Lemmatized_var4', 4:'MDA_Lemmatized_var5', 5:'MDA_Lemmatized_var6', 6:'MDA_Lemmatized_var7', 7:'MDA_Lemmatized_var8', 8:'MDA_Lemmatized_var9', 9:'MDA_Lemmatized_var10'}, inplace=True)

# Create new dataframe for truncated stemmed MDA document term matrix and rename columns to MDA_Stemmed variables
MDA_Stemmed_truncated = pd.DataFrame(document_term_matrix_MDA_Stemmed_truncated)
MDA_Stemmed_truncated.rename(columns={0:'MDA_Stemmed_var1', 1:'MDA_Stemmed_var2', 2:'MDA_Stemmed_var3', 3:'MDA_Stemmed_var4', 4:'MDA_Stemmed_var5', 5:'MDA_Stemmed_var6', 6:'MDA_Stemmed_var7', 7:'MDA_Stemmed_var8', 8:'MDA_Stemmed_var9', 9:'MDA_Stemmed_var10'}, inplace=True)

# Combine all truncated dataframes into one dataframe and called it MDA_SVD
frames = [MDA_truncated, MDA_Lemmatized_truncated, MDA_Stemmed_truncated]
MDA_SVD = pd.concat(frames, axis = 1)

# Merge the copy of the original dataframe MDA_Copy with the truncated dataframe MDA_SVD
frames1 = [MDA_Copy, MDA_SVD]
MDA_Copy1 = pd.concat(frames1, axis = 1)

MDA_Copy1.columns


