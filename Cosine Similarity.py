from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Initialize the object from CountVectorizer class
cv=CountVectorizer();
text = ["London Paris London","Paris Paris London"] #Creating a List of List Items we Have

#Used to count frequency of words in the texts
count_matrix=cv.fit_transform(text)
print("")
print("")
similarity_scores=cosine_similarity(count_matrix)
print(similarity_scores)



