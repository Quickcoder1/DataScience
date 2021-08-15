## Kmeans clustering for EastWestAirlines dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

airlines = pd.read_excel("~/documents/EastWestAirlines.xlsx")
airlines.info()
airlines.describe()
airlines.isna().sum()
plt.hist(airlines)
## we have to normalize the data
def norm_func(i):
    x = (i - i.min())/(i.max() - i.min())
    return(x)
norm_airlines = norm_func(airlines.iloc[:,1:])
norm_airlines

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

TWSS = []
k = list(range(2,12))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(norm_airlines)
    TWSS.append(kmeans.inertia_)
    
TWSS

## screeplot or elbow curve
plt.plot(k, TWSS, "ro-");plt.xlabel("Number_of_clusters");plt.ylabel("total_within_SS")

## selecting 5 clusters from the above scree plot
model = KMeans(n_clusters = 5)
model.fit(norm_airlines)

model.labels_ ## geeting labels of clusters assigned to each row
mb = pd.Series(model.labels_)
mb = pd.DataFrame(mb)
airlines["cluster"] = mb

airlines.head()
airlines.tail()

airlines.iloc[:, 1:].groupby(airlines.cluster).mean()
airlines.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
airlines.to_csv("kmeans for airlines.csv",encoding = "utf-8")
import os
os.getcwd()

### KMeans clustering for Crime data

crime = pd.read_csv("~/documents/crime_data.csv")
crime.isna().sum()

def norm_func(i):
    x = (i - i.min())/(i.max() - i.min())
    return(x)

norm_crime = norm_func(crime.iloc[:,1:5])
norm_crime

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

TWSS = []
k = list(range(1,5))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(norm_crime)
    TWSS.append(kmeans.inertia_)
    
TWSS
### Scree plot or Elbow curve
plt.plot(k, TWSS, "ro-");plt.xlabel("no_of_clusters");plt.ylabel("total_within_SS")

### model 
model = KMeans(n_clusters = 3)
model.fit(norm_crime)

model.labels_
mb = pd.Series(model.labels_)
mb = pd.DataFrame(mb)
crime["clust"] = mb
crime.head()
crime.tail()

crime = crime.iloc[:,[5,0,1,2,3,4]]
crime

norm_crime.to_csv("kmeans for crime.csv")
norm_crime
import os
os.getcwd()

######  KMeans clustering for Insurance data
import pandas as pd
insure = pd.read_csv("~/documents/Insurance Dataset.csv")
insure
insure.isna().sum()

import matplotlib.pyplot as plt
plt.hist(insure)

def norm_func(i):
    x = (i -i.min())/(i.max() - i.min())
    return(x)

norm_insure = norm_func(insure)
plt.hist(norm_insure)

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
TWSS = []

k = list(range(1,5))
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(insure)
    TWSS.append(kmeans.inertia_)

TWSS
### Scree plot or Elbow curve
plt.plot(k, TWSS, "ro-");plt.xlabel("no_of_clusters");plt.ylabel("Total_within_SS")

## taking 3 clusters from the above scree plot

model = KMeans(n_clusters = 3)
model.fit(norm_insure)

## labels for the data
model.labels_
mb = pd.Series(model.labels_)
mb
mb = pd.DataFrame(model.labels_)
mb
insure["cluster"] = mb
insure

insure.iloc[:,0:].groupby(insure.cluster).mean()
insure.iloc[:,[5,0,1,2,3,4]]

insure.to_csv("kmeans for insurance data.csv", encoding = "utf-8")
import os
os.getcwd()
