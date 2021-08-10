import pandas as pd
import numpy as np
data = pd.read_csv("~/documents/calories_consumed.csv")
data
data.info()
data.describe()
data.columns
# measure of central tendency or first moment business decisions
data.mean()
data.median()
data.mode()
## measure of despersion or second business moment decisions
data.std()
data.var()
# third moment business decisions
data.skew()
# fourth moment business decisions
data.kurt()
import matplotlib.pyplot as plt

plt.bar(height = data.weight, x = np.arange(1,15,1))
plt.bar(height = data.calories, x= np.arange(1,15,1))

plt.boxplot(data.weight) ## we have no outliers
plt.boxplot(data.calories) # we have no outliers

plt.hist(data.weight) # the distribution is right skewed
plt.hist(data.calories)# right skewed distribution
import statsmodels.api as sm
sm.qqplot(data)
sm.qqplot(data.weight)
sm.qqplot(data.calories)
# By visualizing histogram,boxplot,qqplots
# we can say the data is normally distributed .
plt.scatter(x = data.weight, y = data.calories, color = 'black')
# correlation 
np.corrcoef(data.weight, data.calories)
# if we have|r| > 0.85, then we can say the co-relation is strong

# also we can say by observing the above scatter plot and correlation
#The data is 1. linearly distributed
             #2. positively distributed
             #3.strong coorelation
             
cov_var = np.cov(data.weight,data.calories)
cov_var

# now we are all set build a model
import statsmodels.formula.api as smf
model = smf.ols("calories~weight", data = data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(data["weight"]))

# regression line
plt.scatter(data.weight, data.calories)
plt.plot(data.weight,pred1,"r")
plt.legend(['Predicted line, Observed data'])
plt.show()

## error calculation
res1 = data.calories - pred1
res_sqr1 = (res1*res1)
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1 ## we got an error = 232.833

##### building a model on transformed data by using log transformations
# x = log(weight); y = calories
plt.scatter(x = np.log(data['weight']), y = data['calories'], color = 'blue')
np.corrcoef(np.log(data.weight),data.calories)

model2 = smf.ols('calories~np.log(weight)',data = data).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(data['weight']))


# regression line
plt.scatter(np.log(data.weight),data.calories)
plt.plot(np.log(data.weight), pred2, 'r')
plt.legend('Predict line', 'Observed data')
plt.show()

## error calculation
res2 = data.calories - pred2
res_sqr2 = (res2 * res2)
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2 # we have an error 253.55

## exponential transformation
# x = weight; y = log(calories)
plt.scatter(x = data['weight'], y = np.log(data['calories']), color = 'orange')
np.corrcoef(data.calories, data.weight)

model3 = smf.ols('np.log(calories)~weight',data = data).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(data['weight']))
pred3_at = np.exp(pred3)
pred3_at

## regression line
plt.scatter(data.weight, np.log(data.calories))
plt.plot(data.weight, pred3, 'r')
plt.legend('Predicted line', 'Observed data')
plt.show()

## error calculation
rse3 = data.calories - pred3_at
rse_sqr3 = (rse3 * rse3)
mse3 = np.mean(rse_sqr3)
rmse3 = np.sqrt(mse3)
rmse3 ### 272.420

#### polynomial transformations
## x = weight; x^2 =(weight * weight); y = log(calories)
model4 = smf.ols('np.log(calories) ~ weight+I(weight*weight)',data = data).fit()
model4.summary()
pred4 = model.predict(pd.DataFrame(data))
pred4_at = np.exp(pred4)
pred4_at

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x = data.iloc[:, 0:1].values
x_poly = poly_reg.fit_transform(x)
x_poly

plt.scatter(data.weight, np.log(data.calories))
plt.plot(x, pred4, color = 'green')
plt.legend('Predicted line','Observed data')
plt.show()

### error calculation
rse4 = data.calories- pred4_at
rse_sqr4 = (rse4 * rse4)
mse4 = np.mean(rse_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

## for choosing best model
new_data = {'model':pd.Series(["SLR","log model","exp model","poly model"]),"RMSE":pd.Series([rmse1,rmse2,rmse3,rmse4])}
table_rmse = pd.DataFrame(new_data)
table_rmse

############################## the best model
from sklearn.model_selection import train_test_split
train,test = train_test_split(data, test_size = 0.2)

final_model = smf.ols('calories~weight',data= train).fit()
final_model.summary()

## predict on test_data
test_pred = final_model.predict(pd.DataFrame(test))
pred_test_calories = np.exp(test_pred)
pred_test_calories

#### model evaluation on test data
test_res = test.calories- pred_test_calories
test_sqrs = (test_res * test_res)
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

### prediction on train data
train_pred = final_model.predict(pd.DataFrame(train))
pred_train_calories = np.exp(train_pred)
pred_train_calories

##### model evaluation on train data
train_res = train.calories - pred_train_calories
train_sqrs = (train_res * train_res)
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse



# 2. A food delivery service recorded the data of  delivery time taken and
# the time taken for the deliveries to be sorted by the restaurants in 
# order to improve their delivery services. Approach – 
# A Simple Linear regression model needs to be built with target
 #variable ‘Delivery.Time’. Apply necessary transformations and
 #record the RMSE values, Correlation coefficient values for different 
 #transformation models. 
 
import pandas as pd
import numpy as np
my_data = pd.read_csv('~/documents/delivery_time.csv')
my_data
my_data.describe()

import matplotlib.pyplot as plt
plt.boxplot(my_data)
plt.boxplot(my_data.delivery)
plt.boxplot(my_data.sorting)

plt.scatter(x = my_data.delivery, y = my_data.sorting, color = "black")
plt.hist(my_data.delivery)
plt.hist(my_data.sorting)

import statsmodels.api as sm
sm.qqplot(my_data.delivery)
sm.qqplot(my_data.sorting)
 ##### by observing the above plots we can say that data is normally distributed
 ## coorelation 
np.corrcoef(my_data.delivery, my_data.sorting)

# covariation
cov_vr = np.cov(my_data.delivery, my_data.sorting)
cov_vr

### model building
import statsmodels.formula.api as smf

mod1 = smf.ols("delivery~sorting",data = my_data).fit()
mod1.summary()
pred1 = mod1.predict(pd.DataFrame(my_data['sorting']))

### regression line
plt.scatter(my_data.sorting, my_data.delivery)
plt.plot(my_data.sorting, pred1,"r")
plt.legend("Predicted line", "Observed data")
plt.show()

## error calculation 
rse1 = my_data.delivery - pred1
rse_sqr1 = (rse1 * rse1)
mse1 = np.mean(rse_sqr1)
rmse1 = np.sqrt(mse1)
rmse1 #### 2.79

### model building on transformed data
## x = log(sorting); y = delivery
plt.scatter(x = my_data.delivery, y = np.log(my_data.sorting))
np.corrcoef(my_data.delivery, np.log(my_data.sorting))


mod2 = smf.ols("delivery~np.log(sorting)",data = my_data).fit()
mod2.summary()
pred2 = mod2.predict(pd.DataFrame(my_data["sorting"]))
### regression line
plt.scatter(np.log(my_data.sorting),my_data.delivery)
plt.plot(np.log(my_data.sorting),pred2, "r")
plt.legend("Predicted line","Observed data")
plt.show()

### error calculation
rse2 = np.log(my_data.sorting) - pred2
rse_sqr2 = (rse2 * rse2)
mse2 = np.mean(rse_sqr2)
rmse2 = np.sqrt(mse2)
rmse2 ### 4.63

### model building on exponential transformations
# x = sorting ; y = log(delivery)
plt.scatter(x = my_data.sorting, y = np.log(my_data.delivery))
np.corrcoef(my_data.sorting, y = np.log(my_data.delivery))

mod3 = smf.ols("np.log(delivery)~sorting",data = my_data).fit()
mod3.summary()
pred3 = mod3.predict(pd.DataFrame(my_data['sorting']))
pred3_at = np.exp(pred3)
pred3_at

### regression line
plt.scatter(my_data.sorting, np.log(my_data.delivery))
plt.plot(my_data.sorting, pred3, "r")
plt.legend('Predicted line',"Observed data")
plt.show()

### error calculation
rse3 = my_data.delivery - pred3_at
rse_sqr3 = (rse3 * rse3)
mse3 = np.mean(rse_sqr3)
rmse3 = np.sqrt(mse3)
rmse3  ### 2.94

##### polynomial  transformation
## x = sorting; x^2 = (sorting*sorting); y = log(delivery)
mod4 = smf.ols("np.log(delivery) ~ sorting + I(sorting*sorting)",data = my_data).fit()
mod4.summary()

pred4 = mod4.predict(pd.DataFrame(my_data))
pred4_at = np.exp(pred4)
pred4_at

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x = my_data.iloc[:, 0:1].values
x_poly = poly_reg.fit_transform(x)

plt.scatter(my_data.sorting, np.log(my_data.delivery))
plt.plot(x, pred4, "r")
plt.legend('Predicted line', 'Observed data')
plt.show()

## error calculation
rse4 = my_data.delivery - pred4_at
rse_sqr4 = (rse4 * rse4)
mse4 = np.mean(rse_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

#### choosing best model
end_data = {"Model": pd.Series(["SLR","log model", "exponential","poly"]), "RMSE": pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(end_data)
table_rmse

### the best model
from sklearn.model_selection import train_test_split

train,test = train_test_split(my_data, test_size = 0.2)
finalmodel = smf.ols('delivery~sorting',data = my_data).fit()
finalmodel.summary()

## predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_delivery = np.exp(test_pred)
pred_test_delivery

### model evaluation on test data
test_rse = test.delivery - test_pred
test_rse_sqr = (test_rse * test_rse)
test_mse = np.mean(test_rse_sqr)
test_rmse = np.sqrt(test_mse)
test_rmse ## 1.5

## prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_del = np.exp(train_pred)
pred_train_del

## model evaluation train data
train_rse = train.delivery - train_pred
train_rse_sqr = (train_rse * train_rse)
train_mse = np.mean(train_rse_sqr)
train_rmse = np.sqrt(train_mse)
train_rmse # 3.08


#3.) A certain organization wanted an early estimate of their employee
#churn out rate. So, the HR department came up with data regarding the 
#employee’s salary hike and churn out rate for a financial year.
#The analytics team will have to perform a deep analysis and predict an
#estimate of employee churn and present the statistics. Approach
#–A Simple Linear regression model needs to be built with target variable
#‘Churn_out_rate’. Apply necessary transformations and record the RMSE values
#, Correlation coefficient values for different transformation.

import pandas as pd 
import numpy as np

info = pd.read_csv('~/documents/emp_data.csv')
info
info.describe()
info.mean()
info.median()
info.mode()
info.std()
info.var()
info.skew()
info.kurt()

import matplotlib.pyplot as plt

### constructing boxplot for finding outliers
plt.boxplot(info)
plt.boxplot(info.salary)
plt.boxplot(info.churn) ## we have no outliers

### check the data is normally distributed or not
import statsmodels.api as sm

sm.qqplot(info.salary)
sm.qqplot(info.churn)

## histogram
plt.hist(info.salary)## right skewed
plt.hist(info.churn)## right skewed distribution

###  Barchart
plt.bar(height = info.salary, x= np.arange(1,11,1))
plt.bar(height = info.churn, x= np.arange(1,11,1))

## scatter plot
plt.scatter(x = info['salary'], y = info['churn'], color = 'red')

### correlation
np.corrcoef(info.salary, info.churn)

## covariance
cov_var = np.cov(info.salary,info.churn)
cov_var

import statsmodels.formula.api as smf
model1 = smf.ols('churn~salary',data= info).fit()
model1.summary()
pred1 = model1.predict(pd.DataFrame(info['salary']))

## regression line
plt.scatter(info.salary, info.churn)
plt.plot(info.salary, pred1, 'r')
plt.legend('Predicted line', 'Observed data')
plt.show()

## error calculation
rse1 = info.churn - pred1
rse_sqr = (rse1 * rse1)
mse1 = np.mean(rse_sqr)
rmse1 = np.sqrt(mse1)
rmse1 ## 3.997

##### model building on log transformed data
# x = log(salary); y = churn
model2 = smf.ols('churn~np.log(salary)',data = info).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(info['salary']))

###  regression line
plt.scatter(np.log(info.salary), info.churn)
plt.plot(np.log(info.salary), pred2, 'r')
plt.legend('Predicted line', 'Observed data')
plt.show()

### error calculation
rse2 = info.churn - pred2
rse_sqr2 = (rse2 * rse2)
mse2 = np.mean(rse_sqr2)
rmse2 = np.sqrt(mse2)
rmse2 ### 3.786

###  model building on exponential data
# x = salary; y = log(churn)
model3 = smf.ols('np.log(churn)~salary',data = info).fit()
model3.summary()
pred3 = model3.predict(pd.DataFrame(info['salary']))
pred3_at = np.exp(pred3)
pred3_at

### regression line
plt.scatter(info.salary, np.log(info.churn))
plt.plot(info.salary, pred3, 'r')
plt.legend('Predicted line', 'Observed data')
plt.show()

#####    Error calculation
rse3 = info.churn - pred3_at
rse_sqr3 = (rse3 * rse3)
mse3 = np.mean(rse_sqr3)
rmse3 = np.sqrt(mse3)
rmse3 #### 3.541

### model building on polynomial transformation
## x = salary+I(salary * salary); y = log(churn)

model4 = smf.ols('np.log(churn)~salary+I(salary * salary)',data = info).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(info))
pred4_at = np.exp(pred4)
pred4_at

### regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x = info.iloc[:,0:1].values
x_poly = poly_reg.fit_transform(x)

### regression line
plt.scatter(info.salary, np.log(info.churn))
plt.plot(x, pred4, 'r')
plt.legend('Predicted line', 'Observed data')
plt.show()

##   Error calculation 
rse4 = info.churn - pred4_at
rse_sqr4 = (rse4 * rse4)
mse4 = np.mean(rse_sqr4)
rmse4 = np.sqrt(mse4)
rmse4 #### 1.326

##### choose the best model using rmse
end_info = {"Model": ['SLR', 'log','exp','poly'], "RMSe": [rmse1, rmse2, rmse3, rmse4]}
table_rmse = pd.DataFrame(end_info)
table_rmse

##### The best model is polynomial model having rmse as 1.326
from sklearn.model_selection import train_test_split
train,test = train_test_split(info, test_size = 0.2)

finalmodel = smf.ols('np.log(info.churn)~salary+I(salary*salary)',data = info).fit()
finalmodel.summary()

###  prediction on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
test_pred_c = np.exp(test_pred)
test_pred_c

### model evaluation
test_rse = test.churn - test_pred_c
test_rse_sqr = (test_rse * test_rse)
test_mse = np.mean(test_rse_sqr)
test_rmse = np.sqrt(test_mse)
test_rmse #### 1.050

### prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
train_pred_c = np.exp(train_pred)
train_pred_c

##### model evaluation 
train_rse = train.churn - train_pred_c
train_rse_sqr = (train_rse * train_rse)
train_mse = np.mean(train_rse_sqr)
train_rmse = np.sqrt(train_mse)
train_rmse ### 1.387


# 4.) The Head HR of a certain organization wants to automate their salary hike 
#estimation. The organization consulted an analytics service provider and 
#asked them to build a basic prediction model by providing them with a sample 
#data that contains historic data of the years of experience and the salary 
#hike given accordingly over the past years. Approach - A Simple Linear 
#regression model needs to be built with target variable ‘Salary’ to predict 
#the salary hikeapply necessary transformations and record the RMSE values, 
#Correlation coefficient values for different transformation models.

import pandas as pd
import numpy as np
data = pd.read_csv('~/documents/Salary_Data.csv')
data
data.columns

import matplotlib.pyplot as plt

#### finding outliers
plt.boxplot(data.YearsExperience)
plt.boxplot(data.Salary)

### check distribution of data
import statsmodels.api as sm
sm.qqplot(data.YearsExperience)
sm.qqplot(data.Salary)

## histogram
plt.hist(data.YearsExperience)
plt.hist(data.Salary)

## barchart
plt.bar(height = data.YearsExperience, x = np.arange(1,31,1))
plt.bar(height = data.Salary, x = np.arange(1,31,1))

### scatter plot
plt.scatter(data.YearsExperience, data.Salary)

### correlation
np.corrcoef(data.YearsExperience, data.Salary)

## covariance
co_var = np.cov(data.YearsExperience, data.Salary)
co_var

import statsmodels.formula.api as smf
m1 = smf.ols('Salary~YearsExperience',data = data).fit()
m1.summary()

p1 = m1.predict(pd.DataFrame(data['YearsExperience']))

#### regression line
plt.scatter(data.YearsExperience, data.Salary)
plt.plot(data['YearsExperience'], p1, 'r')
plt.legend('Predicted line', 'Observed data')
plt.show()

### Error calculation
rse1 = data.Salary - p1
rse_sqr1 = (rse1 * rse1)
mse1 = np.mean(rse_sqr1)
rmse1 = np.sqrt(mse1)
rmse1 ### 5592.043

#### build log transformation model
# x =log(YearsExperience); y = Salary
m2 = smf.ols('Salary~np.log(YearsExperience)', data = data).fit()
m2.summary()

p2 = m2.predict(pd.DataFrame(data['YearsExperience']))

###  Regression line
plt.scatter(np.log(data.YearsExperience), data.Salary)
plt.plot(np.log(data.YearsExperience),p2, 'r')
plt.legend('Predicted line', 'Observed data')
plt.show()

#### Error calculation
rse2 = data.Salary - p2
rse_sqr2 = (rse2 * rse2)
mse2 = np.mean(rse_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

### build exponential transformation model
m3 = smf.ols('np.log(data.Salary)~YearsExperience', data = data).fit()
m3.summary()

p3 = m3.predict(pd.DataFrame(data['YearsExperience']))

##### regression line
plt.scatter(data.YearsExperience, np.log(data.Salary))
plt.plot(data.YearsExperience,p3, 'r')
plt.legend('Predicted line', 'Observed data')
plt.show()

### error calculation
rse3 = data.Salary - p3
rse_sqr3 = (rse3 * rse3)
mse3 = np.mean(rse_sqr3)
rmse3 = np.sqrt(mse3)
rmse3### 80630.25750

#### building model on polynomial transformantions
## y = np.log(Salary); x = YearsExperience; x^2 = (YearsExperience * YearsExperience)
m4 = smf.ols('np.log(data.Salary)~YearsExperience+I(YearsExperience*YearsExperience)',data = data).fit()


p4 = m4.predict(pd.DataFrame(data))
p4_at = np.exp(p4)
p4_at

### Regression line
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)
x = data.iloc[:,0:1].values
x_poly = poly_reg.fit_transform(x)

plt.scatter(data.YearsExperience, data.Salary)
plt.plot(x, p4, 'r')
plt.legend('Predicted line', 'Observed data')
plt.show()

#### Error calculation
rse4 = data.Salary - p4
rse_sqr4 = (rse4 * rse4)
mse4 = np.mean(rse_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

### choosing best model
data_final = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data_final)
table_rmse

##### the best model
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.2)

best_model = smf.ols('Salary~YearsExperience',data = train).fit()
best_model.summary()

### prediction on test data
pred_test = best_model.predict(pd.DataFrame(test))
pred_test_sal = np.exp(pred_test)
pred_test_sal

### Model evaluation on test data
test_rse = test.Salary - pred_test_sal
test_rse_sqr = (test_rse * test_rse)
test_mse = np.mean(test_rse_sqr)
test_rmse = np.sqrt(test_mse)
test_rmse

#### prediction on train data
pred_train = best_model.predict(pd.DataFrame(train))
pred_train_sal = np.exp(pred_train)
pred_train_sal

####  model evaluation on train data
t_rse = train.Salary - pred_train_sal
t_rse_sqr = (t_rse * t_rse)
t_mse = np.mean(t_rse_sqr)
t_rmse = np.sqrt(t_mse)
t_rmse


#A student from a certain University was asked to prepare a dataset and build
# a prediction model for predicting SAT scores based on the exam giver’s GPA. 
#Approach - A regression model needs to be built with target variable 
#‘SAT_Scores’and record the RMSE values, Correlation coefficient values for 
#different transformation models.

import pandas as pd
import numpy as np
mydata = pd.read_csv('~/documents/SAT_GPA.csv')
mydata
mydata.describe()

import matplotlib.pyplot as plt
## boxplot
plt.boxplot(mydata.sat)## no outliers
plt.boxplot(mydata.gpa) ## we have no outliers

###  bar chart
plt.bar(height = mydata.sat, x = np.arange(1,201,1))
plt.bar(height = mydata.gpa, x = np.arange(1,201,1))

## histogram
plt.hist(mydata.sat)
plt.hist(mydata.gpa)

### scatter plot
plt.scatter(mydata.sat, mydata.gpa)

#### lets draw qqplots to check the data normally distributed or not
import statsmodels.api as sm
sm.qqplot(mydata.sat)
sm.qqplot(mydata.gpa)

### corrrelation
np.corrcoef(mydata.sat, mydata.gpa)

## covariance
co_var = np.cov(mydata.sat, mydata.gpa)
co_var

### build model using transformations
import statsmodels.formula.api as smf

m1 = smf.ols('sat~gpa',data = mydata).fit()
m1.summary()

p1 = m1.predict(pd.DataFrame(mydata['gpa']))

## Regression line
plt.scatter(mydata.gpa, mydata.sat)
plt.plot(mydata.gpa, p1, 'r')
plt.legend('Predicted line','Observed data')
plt.show()

#### Error calculation
r = mydata.sat - p1
r_sqr = (r * r)
mse = np.mean(r_sqr)
rmse1 = np.sqrt(mse)
rmse1 ### 166.770

### using log transformations
# y = sat; x = log(gpa)
m2 = smf.ols('sat~np.log(gpa)', data = mydata).fit()
m2.summary()

p2 = m2.predict(pd.DataFrame(mydata['gpa']))

### regression line
plt.scatter(np.log(mydata.gpa), mydata.sat)
plt.plot(np.log(mydata.gpa),p2, 'r')
plt.legend('Predicted line', 'Observed data')
plt.show()

### error calculation
r2 = mydata.sat - p2
r_sqr2 = (r * r)
mse2 = np.mean(r_sqr2)
rmse2 = np.sqrt(mse2)
rmse2 

### using exponential method
# y = log(sat); x = gpa
m3 = smf.ols('np.log(sat)~gpa',data = mydata).fit()
m3.summary()

p3 = m3.predict(pd.DataFrame(mydata['gpa']))
p3_at = np.exp(p3)
p3_at

## Regression line
plt.scatter(mydata.gpa, np.log(mydata.sat))
plt.plot(mydata.gpa, p3, 'r')
plt.legend('Predicted line','Observed data')
plt.show()

### ERROR CALCULATION
rse3 = mydata.sat - p3
rse_sqr3 = (rse3 * rse3)
mse3 = np.mean(rse_sqr3)
rmse3 = np.sqrt(mse3)
rmse3 ### 516.0534

### polynomial transformation
# y = log(sat); x = gpa; x^2 = (gpa * gpa)
m4 = smf.ols('np.log(sat)~gpa+I(gpa*gpa)', data = mydata).fit()
m4.summary()

p4 = m4.predict(pd.DataFrame(mydata))
p4_at = np.exp(p4)
p4_at

### Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x = mydata.iloc[:, 0:1].values
x_ploy = poly_reg.fit_transform(x)

plt.scatter(mydata.gpa, np.log(mydata.sat))
plt.plot(x, p4, 'r')
plt.legend('Predicted line', 'Observed data')
plt.show()

### error calculation
rse4 = mydata.sat - p4_at
rse_sqr4 = (rse4 * rse4)
mse4 = np.mean(rse_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

### choosing best model
finaldata = {"Model":pd.Series(['slr','log','exp','poly']), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(finaldata)
table_rmse

### the best model
best_mod = smf.ols('sat~gpa',data = mydata).fit()
best_mod.summary()

## prediction on test data

from sklearn.model_selection import train_test_split
train, test = train_test_split(mydata, test_size = 0.2)

pred_test = best_mod.predict(pd.DataFrame(test))
pred_test_sat = np.exp(pred_test)
pred_test_sat

### Model evaluation on test data
test_rse = mydata.sat - pred_test_sat
test_rse_sqr = (test_rse * test_rse)
test_mse = np.mean(test_rse_sqr)
test_rmse = np.sqrt(test_mse)
test_rmse

### predction on train data
pred_train = best_mod.predict(pd.DataFrame(train))
pred_train_sat = np.exp(pred_train)
pred_train_sat

### Model evaluation on test data
t_rse = mydata.sat - pred_train_sat
t_rse_sqr = (t_rse * t_rse)
t_mse = np.mean(t_rse_sqr)
train_rmse = np.sqrt(t_mse)
train_rmse
