### categorical data to numerical format by using "One Hot Encoding" and "Label Encoding"


import pandas  as pd
import numpy as np

data = pd.read_csv("~/Documents/animal_category.csv")
data
data.isna().sum()
data.drop(["Index"],axis = 1, inplace = True)
data
df_new = pd.get_dummies(data, drop_first = True)
df_new

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown = "ignore")
enc_data = pd.DataFrame(enc.fit_transform(data).toarray())
enc_data

### label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

data["Animals"] = labelencoder.fit_transform(data["Animals"])
data["Gender"] = labelencoder.fit_transform(data["Gender"])
data["Homly"] = labelencoder.fit_transform(data["Homly"])
data["Types"] = labelencoder.fit_transform(data["Types"])
data
