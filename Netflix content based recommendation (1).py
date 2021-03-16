#!/usr/bin/env python
# coding: utf-8

# # Netflix Recommendation System (Content Based)

# ![image.png](attachment:image.png)

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


netflix_overall=pd.read_csv("netflix_titles.csv")


# The TF-IDF(Term Frequency-Inverse Document Frequency (TF-IDF) ) score is the frequency of a word occurring in a document, down-weighted by the number of documents in which it occurs. This is done to reduce the importance of words that occur frequently in plot overviews and therefore, their significance in computing the final similarity score.

# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[4]:


#removing stopwords
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
netflix_overall['description'] = netflix_overall['description'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(netflix_overall['description'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


# Here, The Cosine similarity score is used since it is independent of magnitude and is relatively easy and fast to calculate.
# 

# ![image.png](attachment:image.png)

# In[5]:


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[6]:


indices = pd.Series(netflix_overall.index, index=netflix_overall['title']).drop_duplicates()


# In[7]:


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return netflix_overall['title'].iloc[movie_indices]


# In[8]:


get_recommendations('Peaky Blinders')


# In[9]:


get_recommendations('Mortel')


# This recommendation is just based on the Plot.

# It is seen that the model performs well, but is not very accurate.Therefore, more metrics are added to the model to improve performance.

# # Content based filtering on multiple metrics

# Content based filtering on the following factors:
# * Title
# * Cast
# * Director
# * Listed in
# * Plot

# Filling null values with empty string.

# In[10]:


filledna=netflix_overall.fillna('')
filledna.head(2)


# In[11]:


def clean_data(x):
        return str.lower(x.replace(" ", ""))


# In[12]:


features=['title','director','cast','listed_in','description']
filledna=filledna[features]


# In[13]:


for feature in features:
    filledna[feature] = filledna[feature].apply(clean_data)
    
filledna.head(2)


# In[14]:


def create_soup(x):
    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']


# In[15]:


filledna['soup'] = filledna.apply(create_soup, axis=1)


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filledna['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[17]:


filledna=filledna.reset_index()
indices = pd.Series(filledna.index, index=filledna['title'])


# In[18]:


def get_recommendations_new(title, cosine_sim=cosine_sim):
    title=title.replace(' ','').lower()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return netflix_overall['title'].iloc[movie_indices]


# In[19]:


get_recommendations_new('PK', cosine_sim2)


# In[20]:


get_recommendations_new('Peaky Blinders', cosine_sim2)


# In[21]:


get_recommendations_new('The Hook Up Plan', cosine_sim2)

