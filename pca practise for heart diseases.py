import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dis = pd.read_csv("~/documents/heart disease.csv")
dis
dis.isna().sum()
plt.hist(dis)

def norm_func(i):
    x = (i - i.min())/(i.max() - i.min())
    return(x)
dis_norm = norm_func(dis)
dis_norm
plt.hist(dis_norm)

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
 
v = linkage(dis_norm, method = "complete", metric = "euclidean")
plt.figure(figsize =(20,19));plt.title("dendro");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(v,
               leaf_rotation =0,
               leaf_font_size =10
               )
plt.show()
## Agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters = 3,linkage = "complete", affinity= "euclidean").fit(dis_norm)
agc.labels_
cluster_labels = pd.Series(agc.labels_)
dis["cluster"] = cluster_labels
dis.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]].groupby(dis.cluster).mean()
dis.to_csv("hcluster for heart.csv",encoding = "utf-8")
import os
os.getcwd()

##### lets consider the clustered data and do pca
hdis = pd.read_csv("C:/Users/Harish/hcluster for heart.csv")
hdis

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
hdis_norm = scale(hdis)
pca = PCA(n_components= 6)
pca_values = pca.fit_transform(hdis_norm)
## pca score
pca_values

## variance
var = pca.explained_variance_ratio_
var
pca.components_
pca.components_[0]
## cumulative variance
var1 = np.cumsum(np.round(var, decimals =4)*100)
var1

## variance plot
plt.plot(var,color = "black")

## dataframe for pca values
pca_data = pd.DataFrame(pca_values)
pca_data
pca_data.columns = "pc0","pc1","pc2","pc3","pc4","pc5"
pca_data
final = pd.concat([hdis.target, pca_data.iloc[:,0:5]],axis = 1)

## scatter plot
plt.scatter(x =final.pc0, y = final.pc1 )
