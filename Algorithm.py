import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#The dataset we are using here is from IMDB dataset of 5000 movies.
#Step1: Read the CSV File.
df = pd.read_csv("movie_dataset.csv")

#print(df.head()) #Printing head of data for veryfying.

#Step2: Select Features:
new_df=df['title']
features=['keywords','cast','genres','director']
for feature in features:
    df[feature] = df[feature].fillna('')

#Step3: Create a column in dataframe which combines all the selected features.
def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row["genres"]+" "+row["director"]
df["combined_features"]=df.apply(combine_features,axis=1) #Passes each row one by one if axis=1
#print(df["combined_features"].head())

#Step 4: Create a count matrix for this new combined column.
cv=CountVectorizer()
count_matrix=cv.fit_transform(df["combined_features"])

#Step 5: Compute the Cosine Similarity based on the Count Matrix.
cosine_sim=cosine_similarity(count_matrix)
movie_user_likes=input("Enter the Movie Name:")

#Step 6: Get index of this movie from its title
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]

movie_index = get_index_from_title(movie_user_likes)
similar_movies=list(enumerate(cosine_sim[movie_index]))

#Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)

## Step 8: Print titles of first 50 movies
i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i=i+1
    if i>50:
        break

##Dump Picke File
pickle.dump(new_df,open('movies.pkl','wb'))
pickle.dump(cosine_sim,open('similarity.pkl','wb'))

