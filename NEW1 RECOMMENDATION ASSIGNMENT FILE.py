#                                  ASSIGNMENT-SOLUTIONS ON RECOMMENDATION


# QUESTION NO.1.SOLUTION:

DataDictionary={"Name of feature":["userId","game","rating"],"Description":["ID of a perticular user","Name of the game","Rating given by user to this game"],"Type":["Quantitative-Nominal","Qualitative-Nominal","Quantitative-Ordinal"],"Relevance":["Irrelevant and does not give much information","Relevant","Relevant"]}
DataDictionary=pd.DataFrame(DataDictionary)
DataDictionary



import pandas as pd
import numpy as np

game = pd.read_csv("~/documents/game.csv",encoding="utf8")

game.shape # shape
game.columns
game.game

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer

game["game"].isnull().sum()
game["game"] = game["game"].fillna(" ")

tfidf_matrix = tfidf.fit_transform(game.game)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

game_index = pd.Series(game.index, index = game['game']).drop_duplicates()

game_id = game_index["Super Mario Galaxy"]
game_id


def get_recommendations(Name, topN):    
    topN = 10
    # Getting the movie index using its title 
    game_id = game_index[Name]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    game_idx  =  [i[0] for i in cosine_scores_N]
    game_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    game_similar_show = pd.DataFrame(columns=["game", "rating"])
    game_similar_show["game"] = game.loc[game_idx, "game"]
    game_similar_show["rating"] = game_scores
    game_similar_show.reset_index(inplace = True)  
    game_similar_show.drop(["index"], axis=1, inplace=True)
    print (game_similar_show)
    # return (anime_similar_show)

    
# Enter your anime and number of anime's to be recommended 
get_recommendations("Super Mario Galaxy", topN = 10)
game_index["Super Mario Galaxy"]




# QUESTION NO.2.SOLUTION:

DataDictionary={"Name of feature":["Id","Titles","Category","Reviews"],"Description":["ID of a perticular user","Title of movie","Category of the movie","Rating given by user to the movie"],"Type":["Quantitative-Nominal","Qualitative-Nominal","Qualitative-Nominal","Quantitative-Ordinal"],"Relevance":["Irrelevant and does not give much information","Relevant","Relevant","Relevant"]}
DataDictionary=pd.DataFrame(DataDictionary)
DataDictionary


import pandas as pd
import numpy as np

movie = pd.read_csv("Entertainment.csv",encoding="utf8")

movie.shape # shape
movie.columns
movie.Category

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer

movie["Category"].isnull().sum()
movie["Catefgory"] = movie["Category"].fillna(" ")

tfidf_matrix = tfidf.fit_transform(movie.Category)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

movie_index = pd.Series(movie.index, index = movie['Titles']).drop_duplicates()

movie_id = movie_index["Toy Story (1995)"]
movie_id


def get_recommendations(Name, topN):    
    topN = 10
    # Getting the movie index using its title 
    movie_id = movie_index[Name]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[movie_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    movie_idx  =  [i[0] for i in cosine_scores_N]
    movie_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    movie_similar_show = pd.DataFrame(columns=["Titles", "rating"])
    movie_similar_show["Titles"] = movie.loc[movie_idx, "Titles"]
    movie_similar_show["rating"] = movie_scores
    movie_similar_show.reset_index(inplace = True)  
    movie_similar_show.drop(["index"], axis=1, inplace=True)
    print (movie_similar_show)
    # return (anime_similar_show)

    
# Enter your anime and number of anime's to be recommended 
get_recommendations("Heat (1995)", topN = 10)
movie_index["Heat (1995)"]
