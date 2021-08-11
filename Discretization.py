### Discretization is basically converting Continuous data to Discrete data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris_data = pd.read_csv("~/Documents/iris.csv")
iris_data
## we have mainly 5 variables named as "sepal.length", sepal.width, "petal.length", "petal.width", "species"
iris_data.isna().sum()
## we have no null values

b1 = [4,4.5,5,5.5,6,6.5,7,7.5,8] ## bins
l1 = [1,2,3,4,5,6,7,8]

iris_data["sl_new"] = pd.cut(iris_data["Sepal.Length"], bins =b1, labels = l1)
print(plt.hist(iris_data["sl_new"]))

## lets consider 2nd variable sepal.width

b2 = [0,2,2.5,3,3.5,4,4.5]
l2 = [0,1,2,3,4,5]

iris_data["sw_new"] = pd.cut(iris_data["Sepal.Width"], bins = b2, labels = l2)
print(plt.hist(iris_data["sw_new"]))

### 3rd variable
b3 = [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7]
l3 = [0,1,2,3,4,5,6,7,8,9,10,11,12]

iris_data["pl_new"] = pd.cut(iris_data["Petal.Length"],bins = b1, labels = l1)
print(plt.hist(iris_data["pl_new"]))

## 4th variable

b4 = [0,0.25,0.5,0.75,1,1.25,1.50,1.75,2,2.25,2.5]
l4 = [1,2,3,4,5,6,7,8,9,10]

iris_data["pw_new"] = pd.cut(iris_data["Petal.Width"], bins = b4, labels = l4)
print(plt.hist(iris_data["pw_new"]))

iris_data = iris_data.replace(to_replace= "Setosa", value = 1)
iris_data = iris_data.replace(to_replace = "versicolor", value = 2)
iris_data = iris_data.replace(to_replace = "virginia", value = 3)

print(iris_data.isna().sum())
final_data = iris_data.drop(["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"], axis = 1)
final_data

