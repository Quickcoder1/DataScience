import pandas as pd
import numpy as np

game = pd.read_csv("~/documents/game.csv", encoding = "utf-8")
game
game.isna().sum()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words ="english")

game["game"].isna().sum()
tfidf_matrix = tfidf.fit_transform(game.game)
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

### computing the cosine similarity for Tfidf_matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

## game_index
game_index = pd.Series(game.index, index = game["game"]).drop_duplicates()
game_index
d
game_id = game_index["Donut County"]
game_id

def get_recommendations(Name, topN):
    topN = 10
    game_id = game_index[Name]
    ##cosine_scores
    cosine_scores = list(enumerate(cosine_sim[game_id]))
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0: topN+1]
    game_idx =  [i[0] for i in cosine_scores_N]
    game_scores = [i[1] for i in cosine_scores_N]
    
    ## game idx and game scores
    game_similar_show = pd.DataFrame(columns=["game", "ratings"])
    game_similar_show["game"] = game.loc[game_idx, "game"]
    game_similar_show["ratings"] = game_scores
    game_similar_show.reset_index(inplace = True)
    game_similar_show.drop(["index"],axis = 1, inplace =True)
    print(game_similar_show)
    
get_recommendations("Donut County", topN = 10)

#### Business-problem -2

##The Entertainment Company, which is an online movie watching platform, wants 
#to improve its collection of movies and showcase those that are highly rated 
#and recommend those movies to its customer by their movie watching footprint. 
#For this, the company has collected the data and shared it with you to provide
#some analytical insights and also to come up with a recommendation algorithm 
#so that it can automate its process for effective recommendations. The ratings 
#are between -9 and +9.

import pandas as pd
import numpy as np

movies = pd.read_csv("~/documents/Entertainment.csv")
movies.columns
movies.isna().sum()

## we have to remove stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words ="english")
movies["Category"].isna().sum()
movies["Category"] = movies["Category"].fillna(" ")

## cosine matrix
tfidf_matrix = tfidf.fit_transform(movies.Category)
tfidf_matrix.shape
# cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)
cosine_sim.shape

movies_index = pd.Series(movies.index, index = movies["Titles"]).drop_duplicates()
## movie_id
movie_id = movies_index["Pocahontas (1995)"]
movie_id

def get_recommendations(Name, TopN):
    TopN = 10
    movie_id = movies_index[Name]
    cosine_scores = list(enumerate(cosine_sim[movie_id]))
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    ## top n cosine_scores
    cosine_scores_N = cosine_scores[0:TopN+1]
    ## movies index
    movie_idx = [i[0] for i in cosine_scores_N]
    movie_scores = [i[1] for i in cosine_scores_N]
    # creating dataframe
    movies_df = pd.DataFrame(columns = ["movie_idx", "movie_name"])
    movies_df["Titles"] = movies.loc[movie_idx, "Titles"]
    movies_df["ratings"] =movie_scores
    movies_df.reset_index(inplace = True)
    movies_df.drop(["index"], axis = 1 , inplace = True)
    print(movies_df)
    
get_recommendations("Heat (1995)", TopN = 10)
movies_index["Heat (1995)"]

### practise -2
import pandas as pd
import numpy as np

cinema = pd.read_csv("~/documents/Entertainment.csv")
cinema
cinema.isna().sum()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = "english")
tfidf_matrix = tfidf.fit_transform(cinema.Category)

from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim_matrix.shape

cinema_index = pd.Series(cinema.index, index = cinema["Titles"]).drop_duplicates()
cinema_id = cinema_index["Heat (1995)"]
cinema_id

def get_recommendations(Name, TopN):
    TopN = 10
    cinema_id = cinema_index[Name]
    cosine_scores = list(enumerate(cosine_sim_matrix[cinema_id]))
    cosine_scores = sorted(cosine_scores,key = lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0:TopN+1]
    movie_idx = [i[0] for i in cosine_scores_N]
    movie_scores = [i[1] for i in cosine_scores_N]
    
    movie_similar_show = pd.DataFrame(columns = ["Titles", "Rattings"])
    movie_similar_show["Title"] = cinema.loc[movie_idx, "Titles"]
    movie_similar_show["Ratings"] = movie_scores
    movie_similar_show.reset_index(inplace = True)
    movie_similar_show.drop(["index"], axis =1, inplace = True)
    print(movie_similar_show)
    
get_recommendations("Heat (1995)", TopN =10)
