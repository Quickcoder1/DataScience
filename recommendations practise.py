import pandas as pd

game = pd.read_csv("~/documents/game.csv",encoding = "utf8")
game.columns
game.info()
game.shape

game.isna().sum()

from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf = TfidfVectorizer(stop_words = 'english')

## tfidf matrix
tfidf_matrix = Tfidf.fit_transform(game.game)
tfidf_matrix.shape
tfidf_matrix

from sklearn.metrics.pairwise import linear_kernel

cosine_sim_matrix = linear_kernel(tfidf_matrix)
game_index = pd.Series(game.index, index = game['game']).drop_duplicates()
game_index

game_id = game_index["Donut County"]
game_id

def get_recommendations(name,topN):
    topN = 10
    game_id = game_index[name]
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse =True)
    cosine_scores_N = cosine_scores[0 :topN+1]
    
    game_idx = [i[0] for i in cosine_scores_N]
    game_scores = [i[1] for i in cosine_scores_N]
    
    #similargames and scores
    game_sim_show = pd.DataFrame(columns = ["game", 'Ratings'])
    game_sim_show["game"] = game.loc[game_idx,'game']
    game_sim_show["Ratings"] = game_scores
    game_sim_show.reset_index(inplace= True)
    print(game_sim_show)
    
get_recommendations('Donut County', topN=10)    


## problem 2

import pandas as pd    

ent = pd.read_csv("~/documents/entertainment.csv", encoding = "utf8")
ent
ent.columns
ent.Titles

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = "english")

ent["Titles"].isna().sum()


tfidf_matrix = tfidf.fit_transform(ent.Titles)
tfidf_matrix

from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix)
cosine_sim_matrix

ent_index = pd.Series(ent.index, index = ent['Titles']).drop_duplicates()
ent_index

ent_id = ent_idx['Lamerica (1994)']
ent_id

def get_recommendations(name,topN):
    topN = 10
    ent_id = ent_index[name]
    cosine_scores = list(enumerate(cosine_sim_matrix[ent_id]))
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1])
    cosine_scores_N = cosine_scores[0:topN+1]
    
    ent_idx =    [i[0] for i in cosine_scores_N]
    ent_scores = [i[1] for i in cosine_scores_N]
    
    ent_sim = pd.DataFrame(columns = ['Titles','scores'])
    ent_sim["name"] = ent.loc[ent_idx, "Titles"]
    ent_sim["scores"] = ent_scores
    
    ent_sim.reset_index(inplace = True)
    print(ent_sim)
    
get_recommendations('Clueless (1995)', topN = 10)
ent_idx['Clueless (1995)']
