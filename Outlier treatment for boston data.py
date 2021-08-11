import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("~/documents/boston_data.csv")
data.isnull()
data.isna().sum()

plt.boxplot(data)
### consider crim variable
plt.boxplot(data["crim"])
## we have an outliers
## we have to remove those outliers by using IQR method
Q1 = data["crim"].quantile(0.25)
Q3 = data["crim"].quantile(0.75)
print(Q1,Q3)

IQR = Q3 - Q1
lw = Q1 - 1.5*IQR
up = Q3 + 1.5*IQR
print(lw, up)

crim_new = pd.DataFrame(np.where(data["crim"]>up, upper_limit, np.where(data["crim"]<lw, lower_limit,data["crim"])))
plt.boxplot(crim_new)

#--------zn---------
plt.boxplot(data['zn']);plt.title('boxplot_zn')
Q1 = data['zn'].quantile(0.25)
Q3 = data['zn'].quantile(0.75)
IQR= Q3-Q1

print(Q1)
print(Q3)
print(IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

print(lower_limit,upper_limit)
data['zn_new']= pd.DataFrame(np.where(data['zn']> upper_limit, upper_limit, np.where(data['zn'] < lower_limit, lower_limit, data['zn'])))
plt.boxplot(data.zn_new);plt.title('boxplot_zn_new');plt.show()

#------indus-----
plt.boxplot(data['indus']);plt.title('Boxplot_indus')

#--------chas--------
plt.boxplot(data['chas']);plt.title('Boxplot_chas')
Q1 = data['chas'].quantile(0.25)
Q2 = data['chas'].quantile(0.75)
IQR= Q2-Q1

print(Q1)
print(Q2)
print(IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q2 + 1.5*IQR

print(lower_limit,upper_limit)
data['chas_new']= pd.DataFrame(np.where(data['chas']> upper_limit, upper_limit, np.where(data['chas'] < lower_limit, lower_limit, data['chas'])))
plt.boxplot(data.chas_new);plt.title('boxplot_chas_new');plt.show()

#------nox------
plt.boxplot(data['nox']);plt.title('Boxplot_nox')


#-------rm------

plt.boxplot(data['rm']);plt.title('Boxplot_rm')
Q1 = data['rm'].quantile(0.25)
Q2 = data['rm'].quantile(0.75)
IQR= Q2-Q1

print(Q1)
print(Q2)
print(IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q2 + 1.5*IQR

print(lower_limit,upper_limit)
data['rm_new']= pd.DataFrame(np.where(data['rm']> upper_limit, upper_limit, np.where(data['rm'] < lower_limit, lower_limit, data['rm'])))
plt.boxplot(data.rm_new);plt.title('boxplot_rm_new');plt.show()

#--------age------
plt.boxplot(data['age']);plt.title('Boxplot_age')

#-------dis-------
plt.boxplot(data['dis']);plt.title('Boxplot_dis')
Q1 = data['dis'].quantile(0.25)
Q2 = data['dis'].quantile(0.75)
IQR= Q2-Q1

print(Q1)
print(Q2)
print(IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q2 + 1.5*IQR

print(lower_limit,upper_limit)
data['dis_new']= pd.DataFrame(np.where(data['dis']> upper_limit, upper_limit, np.where(data['dis'] < lower_limit, lower_limit, data['dis'])))
plt.boxplot(data.dis_new);plt.title('boxplot_dis_new');plt.show()

#-------rad-------
plt.boxplot(data['rad']);plt.title('Boxplot_rad')

#-------tax-------
plt.boxplot(data['tax']);plt.title('Boxplot_tax')

#---------ptratio-------
plt.boxplot(data['ptratio']);plt.title('Boxplot_ptratio')
Q1 = data['ptratio'].quantile(0.25)
Q2 = data['ptratio'].quantile(0.75)
IQR= Q2-Q1

print(Q1)
print(Q2)
print(IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q2 + 1.5*IQR

print(lower_limit,upper_limit)
data['ptratio_new'] = pd.DataFrame(np.where(data['ptratio']> upper_limit,upper_limit, np.where(data['ptratio'] < lower_limit, lower_limit,data['ptratio'])))
plt.boxplot(data.ptratio_new);plt.title('boxplot_ptratio_new')


#------black--------

plt.boxplot(data['black']);plt.title('boxplot_black')
Q1 = data['black'].quantile(0.25)
Q2 = data['black'].quantile(0.75)
IQR= Q2-Q1

print(Q1)
print(Q2)
print(IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q2 + 1.5*IQR

print(lower_limit,upper_limit)
data['black_new']=pd.DataFrame(np.where(data['black']> upper_limit, upper_limit, np.where(data['black'] < lower_limit, lower_limit ,data['black'])))
plt.boxplot(data.black_new);plt.title('boxplot_black_new')

#-------lstat------
plt.boxplot(data['lstat']);plt.titile('Boxplot_lstat')
Q1 = data['lstat'].quantile(0.25)
Q2 = data['lstat'].quantile(0.75)
          
IQR= Q2-Q1

print(Q1)
print(Q2)
print(IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q2 + 1.5*IQR

print(lower_limit,upper_limit)
data['lstat_new']=pd.DataFrame(np.where(data['lstat']> upper_limit, upper_limit,np.where(data['lstat'] < lower_limit, lower_limit , data['lstat'])))
plt.boxplot(data.lstat_new);plt.title('Boxplot_lstat_new')

#-------medv-----
plt.boxplot(data['medv']); plt.title('Boxplot_medv')
Q1 = data['medv'].quantile(0.25)
Q2 = data['medv'].quantile(0.75)
IQR= Q2-Q1

print(Q1)
print(Q2)
print(IQR)

lower_limit = Q1 - 1.5*IQR
upper_limit = Q2 + 1.5*IQR

print(lower_limit,upper_limit)
data['medv_new'] = pd.DataFrame(np.where(data['medv'] > upper_limit, upper_limit, np.where(data['medv'] < lower_limit, lower_limit, data['medv'])))
plt.boxplot(data.medv_new); plt.title('Boxplot_medv_new')

