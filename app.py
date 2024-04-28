import streamlit as st 
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv('tags.csv')
cv=CountVectorizer(stop_words='english',max_features=5000)
vectors=cv.fit_transform(df['tags']).toarray()
similarity=cosine_similarity(vectors)

def fetch_poster(movie_id):
    response=requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US")
    data=response.json()
    return "https://image.tmdb.org/t/p/w500"+data['poster_path']

similarity=np.array(similarity)

def recommend(movie):
    ind=df[df['title']==movie].index[0]
    distances=similarity[ind]
    ans_ind=distances.argsort()[-6:-1]
    y=df[['id','title']].iloc[ans_ind].values
    return np.flip(y,axis=0)

st.title("Movie Recommender")
selected=st.selectbox("Select a movie",df['title'])
if st.button('Recommend'):
    movies=recommend(selected)
    cols=st.columns(5)
    for i in range(5):
        with cols[i]:
            st.write(movies[i][1])
            st.image(fetch_poster(movies[i][0]))
# print(option)