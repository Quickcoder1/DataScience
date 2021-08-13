
###1.)Perform clustering for the airlines data to obtain optimum number of clusters
#Draw the inferences from the clusters obtained. Refer to EastWestAirlines.xlsx
#dataset.

import pandas as pd
import matplotlib.pylab as plt

input = pd.read_excel("~/Documents/EastWestAirlines.xlsx")
input
new_data = input.rename(columns={"ID#":"id","Award?":"Award"})
new_data
new_data.isna().sum()

def norm_func(i):
    x= i-i.min()/i.max()-i.min()
    return(x)

df_norm = norm_func(new_data.iloc[:,1:])
df_norm
df_norm.describe()
df_norm.columns
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(df_norm, method = "complete", metric ="euclidean")
plt.figure(figsize=(10, 5));plt.title('Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')

sch.dendrogram(z,
    leaf_rotation=0,
    leaf_font_size=10,
)
plt.show()

## we have to do agglomerative clustering to choose no.of clusters from above dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=4,affinity="euclidean", linkage= "complete").fit('df_norm')         
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
new_data["clust"] = cluster_labels
input = new_data.iloc[:,[0,12,1,2,3,4,5,6,7,8,9,10,11]]
input.head()       
input.iloc[:, 2:].groupby(new_data.clust).mean()
new_data.to_csv("EastWestAirlines.csv", encoding = "utf-8")

# 2)Perform clustering for the crime data and identify the number of clusters 
# formed and draw inferences. Refer to crime_data.csv dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
crime_data = pd.read_csv("~/documents/crime_data.csv")
crime_data
crime_data.isnull()
crime_data.isna().sum()
crime_data.describe()
new_data = crime_data.iloc[:,1:5]
new_data

## we have to normalize the data here

def norm_func(i):
    x =i -i.min()/i.max()-i.min()
    return(x)

d_norm = norm_func(new_data)
d_norm
d_norm.columns

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(d_norm, method = "complete", metric = "euclidean")

# creating a dendrogram
plt.figure(figsize=(12,8));plt.title("dendrogram");plt.xlabel("Index");plt.ylabel("distance")
sch.dendrogram(z,
               leaf_rotation = 10,
               leaf_font_size = 2
               )
plt.show()

from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 4, linkage ="complete", affinity="euclidean").fit(d_norm)
h_complete
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
cluster_labels
new_data["clust"] = cluster_labels
new_data
crime_data = new_data.iloc[:, [4,0,1,2,3]]
crime_data
crime_data.head()
crime_data
## Aggregate mean of each cluster

crime_data.iloc[:,1:].groupby(crime_data.clust).mean()
# creating a csv file

crime_data.to_csv("crime_data", encoding = "utf8")
import os
os.getcwd()

### 3.Perform clustering analysis on the telecom data set. 
# The data is a mixture of both categorical and numerical data.
# It consists the number of customers who churn. 
# Derive insights and get possible information on factors that may affect
# the churn decision. Refer to Telco_customer_churn.xlsx dataset.
#Hint: 
# •	Perform EDA and remove unwanted columns.
# •	Use Gower dissimilarity matrix, In R use daisy() function.
import pandas as pd
telco = pd.read_excel("~/documents/Telco_customer_churn.xlsx")
telco
telco.columns
telco.describe()
telco.shape
new_telco = telco.drop(["Customer ID","Count", "Quarter","Referred a Friend","Number of Referrals","Offer","Online Security","Online Backup","Paperless Billing","Payment Method"], axis = 1)
b = new_telco.iloc[:,:]
m = pd.DataFrame(b)
new_telco
new_telco.describe()
new_telco.columns

from sklearn.preprocessing import LabelEncoder
en_data = new_telco.apply(LabelEncoder().fit_transform)
en_data
en_data.columns
## to convert categorical data to numeric data we should create dummies

def norm_func(i):
    
    x = (i-i.min()/i.max()-i.min())
    return(x)

final_data = norm_func(en_data)
final_data

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt
z = linkage(final_data, method= "complete", metric = "Euclidean")
## creating dendrogram for visualization of clustered data
plt.figure(figsize=(16,10));plt.title =("dendrogram");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(z,
               leaf_rotation = 10,
               leaf_font_size = 2
               )
plt.show()
# Now applying AgglomerativeClustering choosing 4 as a clusters from above dendrogram

from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = "complete",affinity = "Euclidean").fit(final_data)
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
final_data["clust"]= cluster_labels
final_data
telco = final_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
                           20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]]
telco
telco.head()
telco.columns
telco.to_csv("Telco_customer_churn.xlsx", encoding = "utf-8")
import os
os.getcwd()

## 4.)Perform clustering on mixed data convert the categorical variables to 
# numeric by using dummies or Label Encoding and perform normalization 
# techniques. The data set consists details of customers related to auto 
# insurance. Refer to Autoinsurance.csv dataset.
import pandas as pd

df = pd.read_csv("~/documents/AutoInsurance.csv")
df
df.columns
df.shape
x = df.iloc[:,:]
y = pd.DataFrame(x)

# applying LabelEncoder for converting categorical data to numerical data

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
enc_data = (df.apply(LabelEncoder().fit_transform))
enc_data 
enc_data.head()
## we should normalize the data
def norm_func(i):
    x =(i-i.min()/i.max()-i.min())
    return(x)
new_norm = norm_func(enc_data)
new_norm
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
k = linkage(new_norm, method = "complete", metric ="euclidean")
# dendrogram
import matplotlib.pylab as plt
plt.figure(figsize=(18,10)); plt.title ="dendrogram"; plt.xlabel("Index"); plt.ylabel("Distance")
sch.dendrogram(z,
               leaf_rotation = 10,
               leaf_font_size = 0
               )
plt.show()

## performing agglomerativeclustering for choosing n cluster from above dendrogram
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters= 5, linkage ="complete",affinity= "euclidean").fit(enc_data)
h_complete
h_completelabels_
cluster_labels = pd.Series(h_complete.labels_)
new_norm["clust"] = cluster_labels
new_norm = df.iloc[:,23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                   21,22]
new_norm.head()
new_norm.to_csv("AutoInsurance.csv")
import os
os.getcwd()
