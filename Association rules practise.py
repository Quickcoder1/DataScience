#### Association rules for books data

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
books = pd.read_csv("~/documents/book.csv")
books.isna().sum()
frequent_items = apriori(books,min_support = 0.8,use_colnames = True)
frequent_items

#### items frequency
freq_items = apriori(books, min_support= 0.1, use_colnames = True)
freq_items
## rules
rules = association_rules(freq_items,metric ="lift",min_threshold = 1)
rules
rules = rules.sort_values(["confidence","lift"],ascending=[False,False])
rules


## association rules for movies data
movies = pd.read_csv("~/documents/my_movies.csv")
movies
movies.head()
movies_new = movies.iloc[:,5:]
movies_new

from mlxtend.frequent_patterns import apriori, association_rules
frequent_patterns = apriori(movies_new, min_support=0.7, use_colnames =True)
frequent_patterns

## frequent items
freq_items = apriori(movies_new, min_support = 0.1, use_colnames =True)
freq_items

rules = association_rules(freq_items, metric="lift",min_threshold =1)
rules
rules = rules.sort_values(["confidence","lift"],ascending =[ False, False])
rules

## association rules for phonedata
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

phone = pd.read_csv("~/documents/myphonedata.csv")
phone
mobile = phone.iloc[:,3:]
mobile
mobile.isna().sum()

from mlxtend.frequent_patterns import apriori, association_rules
freq_patterns = apriori(mobile, min_support= 0.8, use_colnames =True)
freq_patterns

## frequency items
freq_items = apriori(mobile, min_support=0.1, use_colnames = True)
freq_items

rules = association_rules(freq_items,metric ="lift",min_threshold =1)
rules
rules = rules.sort_values(["confidence","lift"],ascending=[False,False])


### association rules for groceries data
groceries = []
with open("C:/Users/Harish/Documents/groceries.csv") as f:
    groceries = f.read()
    
groceries
### split the data
groceries = groceries.split("\n")
groceries

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    
all_groceries_list = [i for item in groceries_list for i in item]
from collections import Counter
item_frequencies = Counter(all_groceries_list)
item_frequencies

## sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])
item_frequencies

### storing items and frequencies in separate variables
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

## barplot for top 10
import matplotlib.pyplot as plt
plt.bar(height = frequencies[0:10],x = list(range(0,10)), color = "red")
plt.xticks(list(range(0,10)), items[0:10])
plt.xlabel("Items")
plt.ylabel("count")
plt.show()

## creating DataFrame for the transaction data
groceries_df = pd.DataFrame(pd.Series(groceries_list))
groceries_df
## remove last empty transaction
groceries_df = groceries_df.iloc[:9835,:]
groceries_df.columns = ["transactions"]

X = groceries_df["transactions"].str.join(sep ="*").str.get_dummies(sep = "*")
X

from mlxtend.frequent_patterns import apriori, association_rules
frequent_items = apriori(X, min_support = 0.0075, max_len =4, use_colnames = True)
frequent_items

frequent_items.sort_values("support", ascending = False, inplace = True)

### most frequent item sets based on support
plt.bar(height = frequent_items.support[0:15], x = list(range(0,15)),color = "rgymk")
plt.xticks(list(range(0,15)), rotation =40)
plt.xlabel("items")
plt.ylabel("support")
plt.show()

## rules 
rules = association_rules(frequent_items, metric="lift", min_threshold = 1)
rules
rules.sort_values("lift", ascending = False).head()

def to_list(i):
    return(sorted(list(i)))

ma_X = rules.antecedents.apply(to_list)+ rules.consequents.apply(to_list)
ma_X  = ma_X.apply(sorted)
rules_set = list(ma_X)
rules_set

unique_rules_set = [list(m) for m in set(tuple(i) for i in rules_set)]

index_rules = []

for i in unique_rules_set:
    index_rules.append(rules_set.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)

rules_no_redudancy
rules_no_redudancy.to_csv("associations for groceries.csv",encoding = "utf-8")
import os
os.getcwd()

##   Question - 5

#A retail store in India, has its transaction data, and it would like to know
# the buying pattern of the 
#consumers in its locality, you have been assigned this task to provide the
# manager with rules 
#on how the placement of products needs to be there in shelves so that it can
# improve the buying
#patterns of consumes and increase customer footfall

transactions = []
with open("C:/Users/Harish/Documents/transactions_retail1.csv") as f:
    transactions = f.read()
    
transactions

## we have to split the data
transactions = transactions.split("\n")
transactions = transactions[0:2000]
transactions_list = []

## we have to create transactions list
for i in transactions:
    transactions_list.append(i.split(","))
    
all_transactions_list = [i for item in transactions_list for i in item]
all_transactions_list

from collections import Counter

item_frequencies = Counter(all_transactions_list)
item_frequencies

#### practise - 2
products = []
with open("C:/Users/Harish/Documents/transactions_retail1.csv") as f:
    products = f.read()
    
products
products = products.split("\n")
products
### we have to make a list 
products_list = []
for i in products:
    products_list.append(i.split(","))
    
all_products_list = [i for item in products_list for i in item]
all_products_list

from collections import Counter
item_frequencies = Counter(all_products_list)
item_frequencies

#### we have to sort the data
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])
item_frequencies

### storing frequencies and items in separate variable
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

## Barplot for top 10 transactions
import matplotlib.pyplot as plt
plt.bar(height = frequencies[0:10], x = list(range(0,10)), color = "pink")

### creating DataFrame for Products data
import pandas as pd
products_df = pd.DataFrame(pd.Series(products_list))
products_df
products_df = products_df.iloc[:557040,:]
products_df
### we have to include the column named "Transactions"
products_df.columns = ["Transactions"]

### creating a dummy column for each item in each transactions
X = products_df["Transactions"].str.join(sep = "*").str.get_dummies(sep = "*")
X
