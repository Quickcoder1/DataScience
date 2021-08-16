import pandas as pd 
import numpy as np


DataDictionary={"Name of feature":["Type","Alcohol","Malic","Ash","Alcalinity","Magnesium","Phenols","Flavanoids","Nonflavanoids","Proanthocyanins","Color","Hue","Dilution","Proline"],"Description":["Type of wine","Amount of Alcolhol","Amount of Malic","Amount of Ash","Amount of Alcalinity","Amount of Magnesium","Amount of phenols","Amount of flavanoida","Amount of Nonflavanoids","Amount of Proanthocyanins","Amount of Colour","Amount Of Hue","Amount of Dilution","Amount of Proline"],"Type":["Quantitative-Nominal","Quantitative","Quantitative","Quantitative","Quantitative","Quantitative","Quantitative","Quantitative","Quantitative","Quantitative","Quantitative","Quantitative","Quantitative","Quantitative"],"Relevance":["Relevant but does not give much information","Relevant","Relevant","Relevant","Relevant","Relevant","Relevant","Relevant","Relevant","Relevant","Relevant","Relevant","Relevant","Relevant"                                                                                                            ]}
DataDictionary=pd.DataFrame(DataDictionary)
DataDictionary



data1 = pd.read_csv("~/documents/wine.csv")
data1.describe()

data = data1.drop(["Type"], axis = 1)
data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
data = data.iloc[:, 1:13]

# Normalizing the numerical data 
data_normal = scale(data)
data_normal

pca = PCA(n_components = 10)
pca_values = pca.fit_transform(data_normal)
pca_values
# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

pca.components_
pca.components_[0]
# Cumulative variance 

var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")
plt.show()
# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5","comp6","comp7","comp8","comp9"                                          
final = pd.concat([data1.Type, pca_data.iloc[:, 0:3]], axis = 1)

# Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = final.comp0, y = final.comp1)
plt.show()



# now we will cluster this data using hclustering.

data2 = final.iloc[:,1:]
data2.describe()
data2.info()


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(data2, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(data2) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

data2['clust'] = cluster_labels # creating a new column and assigning it to new column 

data2 = data2.iloc[:, [3,0,1,2]]
data2.head()
data2.tail()
# Aggregate mean of each cluster
data2.iloc[:, :4].groupby(data2.clust).mean()

# creating a csv file 
data2.to_csv("pcahclust.csv", encoding = "utf-8")

# now we will perform kmeans clustering on our final data of pca.
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 


data2 = final.iloc[:,1:]
data2.describe()
data2.info()

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data2)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(data2)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data2['clust'] = mb # creating a  new column and assigning it to new column 

data2.head()
data2.tail()

data2 = data2.iloc[:,[3,0,1,2]]
data2.head()
data2.tail()
data2.iloc[:, 1:].groupby(data2.clust).mean()

data2.to_csv("pcaKmeans.csv", encoding = "utf-8")



# QUESTION NO.2 SOLUTION:

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 


data3=pd.read_csv("heart disease.csv")
data3.describe()
data3.info()
data4=data3.iloc[:,:13]
scaleddata=scale(data4)

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(scaleddata, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 6 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 6, linkage = 'complete', affinity = "euclidean").fit(scaleddata) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

data3['clust'] = cluster_labels # creating a new column and assigning it to new column 

data3 = data3.iloc[:, [14,13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
data3.head()
data3.tail()
# Aggregate mean of each cluster
data3.iloc[:, 2:].groupby(data3.clust).mean()

# creating a csv file 
data3.to_csv("beforepcapy.csv", encoding = "utf-8")

# PC formation

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

pca = PCA(n_components =13 )
pca_values = pca.fit_transform(scaleddata)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

pca.components_
pca.components_[0]
# Cumulative variance 

var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")
plt.show()
# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5","comp6","comp7","comp8","comp9","comp10","comp11","comp12"
final = pd.concat([data3.target, pca_data.iloc[:, 0:9]], axis = 1)

# Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = final.comp0, y = final.comp1)
plt.show()


# After pca dendogram

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

data5=final.iloc[:,1:]
z = linkage(data5, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(data5) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

data5['clust'] = cluster_labels # creating a new column and assigning it to new column 

data5 = data5.iloc[:, [9,0,1,2,3,4,5,6,7,8]]
data5.head()
data5.tail()
# Aggregate mean of each cluster
data5.iloc[:, 1:].groupby(data5.clust).mean()

# creating a csv file 
data5.to_csv("afterpcapy.csv", encoding = "utf-8")











































































































