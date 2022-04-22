import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movie_dataset.csv")

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

def recommend():
    # Step2: Select Features:
    new_df = df['title']
    features = ['keywords', 'cast', 'genres', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')

    # Step3: Create a column in dataframe which combines all the selected features.
    def combine_features(row):
        return row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + row["director"]

    df["combined_features"] = df.apply(combine_features, axis=1)  # Passes each row one by one if axis=1
    # print(df["combined_features"].head())

    # Step 4: Create a count matrix for this new combined column.
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])

    # Step 5: Compute the Cosine Similarity based on the Count Matrix.
    cosine_sim = cosine_similarity(count_matrix)
    movie_user_likes = option

    movie_index = get_index_from_title(movie_user_likes)
    similar_movies = list(enumerate(cosine_sim[movie_index]))

    # Step 7: Get a list of similar movies in descending order of similarity score
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    ## Step 8: Print titles of first 50 movies
    i = 0
    for movie in sorted_similar_movies:
        st.text(get_title_from_index(movie[0]))
        i = i + 1
        if i > 10:
            break

st.image("Header.jpg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.title("Movie Recommender")
movies_list=pickle.load(open('movies.pkl','rb'))

option=st.selectbox('Movies',movies_list)

if st.button('Recommend'):
    recommend()

