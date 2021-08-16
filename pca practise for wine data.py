import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("~/documents/wine.csv")
data
data.isna().sum()
data.drop(columns = ("Type"),inplace = True)
data
plt.hist(data)
## we have to normalize the data
def norm_func(i):
    x = (i -i.min())/(i.max() - i.min())
    return(x)
norm_data = norm_func(data)

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(norm_data, method = "complete", metric ="euclidean")
plt.figure(figsize=(20,10));plt.title("dendrogram");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(z,
               leaf_rotation = 10,
               leaf_font_size = 10
               )
plt.show()

### agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters= 4, linkage= "complete", affinity = "euclidean").fit(norm_data)
ac.labels_
cluster_labels = pd.Series(ac.labels_)
cluster_labels
data["cluster"] = cluster_labels
data

data.iloc[:,1:].groupby(data.cluster).mean()
data.iloc[:,[13,0,1,2,3,4,5,6,7,8,9,10,11,12]].groupby(data.cluster).mean()
data
data.to_csv("hcluster for wine.csv",encoding = "utf-8")
import os
os.getcwd()

#### performing pca on the cluster data
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

Hclust = pd.read_csv("C:/Users/Harish/hcluster for wine.csv")
Hclust
Hclust.isna().sum()
Hclust.rename(columns = {"Unnamed: 0": "unnamed"},inplace = True)
Hclust
clust = Hclust.drop(["unnamed"],axis = 1)
clust
### now we have to normalize the data 
clust_norm = scale(clust)
clust_norm
plt.hist(clust_norm)

pca = PCA(n_components = 3)
pca_values = pca.fit_transform(clust_norm)
pca_values
## variance between pca's
var = pca.explained_variance_ratio_
var

pca.components_
pca.components_[0]
## cumulative variance
var1 = np.cumsum(np.round(var,decimals = 4)* 100)
var1
## variance plot 
plt.plot(var1,color= "red")

## pca score
pca_values
pca_data = pd.DataFrame(pca_values)
pca_data
pca_data.columns = "comp1","comp2","comp3"
pca_data
final  = pd.concat([Hclust.unnamed, pca_data.iloc[:,0:3]],axis = 1)

## scatter plot
plt.scatter(x =final.comp1 , y =final.comp2)
plt.scatter(x= final.comp2, y = final.comp3)

final.to_csv("pcaclust for wine.csv",encoding = "utf-8")
import os
os.getcwd()

### clustering on pca data
pc = pd.read_csv("C:/Users/Harish/pcaclust for wine.csv")
pc
dr = pc.drop(["Unnamed: 0", "unnamed"],axis = 1)
dr
plt.hist(dr)
def norm_func(i):
    x = (i -i.min())/(i.max() - i.min())
    return(x)
dr_norm = norm_func(dr)
dr_norm
plt.hist(dr_norm)

from scipy.cluster.hierarchy import linkage
x = linkage(dr_norm, method ="complete", metric = "euclidean")
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(20,10));plt.title("dendrogram");plt.xlabel("index");plt.ylabel("distance")
sch.dendrogram(x,
               leaf_rotation = 0,
               leaf_font_size = 10
               )
plt.show()
#### agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
ag = AgglomerativeClustering(n_clusters = 3, linkage = "complete", affinity = "euclidean").fit(dr_norm)
ag.labels_
cluster_labels = pd.Series(ag.labels_)
dr["clust"] = cluster_labels
dr
dr.iloc[:,[3,0,1,2]].groupby(dr.clust).mean()
dr.to_csv("pcafinal.csv",encoding = "utf-8")

import os
os.getcwd()
