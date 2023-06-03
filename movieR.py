#!/usr/bin/env python
# coding: utf-8

# # Content based Recommender System

# based on search history

# In[46]:


import numpy as np
import pandas as pd
import ast


# In[47]:


movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')


# # merging movies on the basis of "Title"

# In[48]:


movies = movies.merge(credits,on = 'title')


# # column ki chhatni kr denge

# In[49]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# # delete null columns

# In[50]:


movies.isnull().sum()


# In[51]:


movies.dropna(inplace = True)


# # checking duplicate data

# In[52]:


movies.duplicated().sum()   


# In[53]:


movies.iloc[0].genres


# In[54]:


ast.literal_eval(movies.iloc[0].genres)


# In[55]:


def convert(a):               # a is string here
    l=[]
    for i in ast.literal_eval(a):    # a becomes list
        l.append(i['name'])
    return l    


# In[56]:


movies['genres'][0]


# In[57]:


movies['keywords'][0]


# In[58]:


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies.head(1)


# # extracting top 4 actors

# In[59]:


def convert2(a):             
    l=[]
    count = 0
    for i in ast.literal_eval(a):
        if count==4:
            break
        l.append(i['name'])
        count+=1
    return l    


# In[60]:


movies['cast'][0]


# In[61]:


movies['cast'] = movies['cast'].apply(convert2)
movies.head(1)


# # fetch director from crew

# In[62]:


def fetch_director(a):             
    l=[]
    for i in ast.literal_eval(a):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l    


# In[63]:


movies['crew'][0]


# In[64]:


movies['crew'] = movies['crew'].apply(fetch_director)
movies.head(1)


# # converting overview in list

# In[65]:


movies['overview'][0]


# In[66]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.head(1)


# # removing space from full name

# In[67]:


movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])
movies.head(5)


# # attach columns in a single tag

# In[68]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head(5)


# In[69]:


new_movies = movies[['movie_id','title','tags']]
new_movies.head(5)


# # (converting tags into string)

# In[70]:


new_movies['tags'] = new_movies['tags'].apply(lambda x:" ".join(x))
new_movies['tags'][0]


# In[71]:


new_movies['tags'] = new_movies['tags'].apply(lambda x:x.lower())
new_movies['tags'][0]


# # converting like actions and action into 1 action only

# In[72]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[73]:


def stem(text):
    l=[]
    for i in text.split():       # converting text into list
        l.append(ps.stem(i))
    return " ".join(l)           # converting back to string


# In[74]:


new_movies['tags'] = new_movies['tags'].apply(stem)
new_movies['tags'][0]


# # Vectorisation (text -> vectors) using 'bag of words' technique
# adding all tags and select 5000 common words

# In[75]:


from sklearn.feature_extraction.text import CountVectorizer as cvz
cv = cvz(max_features = 3000, stop_words='english')


# In[76]:


vectors = cv.fit_transform(new_movies['tags']).toarray()
vectors


# In[77]:


cv.get_feature_names_out()


# # calculate cosine distance between every movie using tags

# In[78]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
similarity


# # fetching similar movies

# In[79]:


sorted(list(enumerate(similarity[0])),reverse = True, key = lambda x:x[1])[1:6]


# In[80]:

def recommend(movie):
    movie_index = new_movies[new_movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = [new_movies.iloc[i[0]].title for i in movie_list]
    return recommended_movies








"""

def recommend(movie):
    movie_index = new_movies[new_movies['title']==movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse = True, key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_movies.iloc[i[0]].title)

"""

# In[87]:


#recommend('Home')


# In[88]:


#recommend('Planet 51')


# # sending this code to pycharm

# In[83]:


#import pickle


# In[84]:


#pickle.dump(new_movies.to_dict(),open('movies_dict.pkl','wb'))


# In[85]:


#pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




