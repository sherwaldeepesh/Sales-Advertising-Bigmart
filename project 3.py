#Task1(Part 1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
import statsmodels.api as sm

df_marketing = pd.read_csv('SalesBasedOnAdvertising.csv')

#Task1(Part 2)
df_marketing.shape

#Task1(Part 3)
df_marketing.head()
df_marketing.describe()
df_marketing.dtypes

#Task 1(Part 4)
df_marketing.isnull().sum()

df_marketing[df_marketing['sales']==0]

df_marketing=df_marketing.drop((df_marketing[df_marketing['sales']==0]).index)

df_marketing['radio'] = df_marketing['radio'].fillna(df_marketing['radio'].mean())

df_marketing['sales'] = df_marketing['sales']*100
#Task 1(Part 5)
df_marketing.shape

df_marketing.head()
df_marketing.describe()
df_marketing.dtypes

#Task 2(Part 1)
fig = plt.figure(figsize = (5,12))
ax1 = plt.subplot(411)
ax1 = plt.boxplot(df_marketing['TV'])
ax1 = plt.title('Boxplot of Marketing[TV]')

ax2 = plt.subplot(412)
ax2 = plt.boxplot(df_marketing['radio'])
ax1 = plt.title('Boxplot of Marketing[Radio]')

ax3 = plt.subplot(413)
ax3 = plt.boxplot(df_marketing['newspaper'])
ax1 = plt.title('Boxplot of Marketing[Newspaper]')

ax4 = plt.subplot(414)
ax4 = plt.boxplot(df_marketing['sales'])
ax1 = plt.title('Boxplot of Marketing[Sales]')

#Task 2(Part 2)
fig = plt.figure(figsize = (13,6))
ax1 = plt.subplot(211)
ax1 = plt.title('Boxplot of Advertising[TV,Radio,Newspaper] and Sales')
ax1 = plt.boxplot([df_marketing['sales'],df_marketing['TV'],df_marketing['radio'],df_marketing['newspaper']])
ax1 = plt.xticks([1,2,3,4],['Sales','TV','Radio','Newspaper'])


ax2 = plt.subplot(212)
ax2 = plt.title('Plot of Advertising[TV,Radio,Newspaper] and Sales', pad = 2)
ax2 = plt.plot(df_marketing['sales'],label = 'Sales')
ax2 = plt.plot(df_marketing['TV'],label = 'TV')
ax2 = plt.plot(df_marketing['radio'],label = 'Radio')
ax2 = plt.plot(df_marketing['newspaper'],label = 'Newspaper')
ax2 = plt.ylabel('Amount thousands for advt & lakhs for sales ')
plt.legend(loc = 'best')

#Task 3(Part 1)

#Task 3(Part 2)
sns.distplot(df_marketing['sales'])

#Task 4(Part 1)
x1 = df_marketing['TV'].values.reshape(-1,1)
y1 = df_marketing['sales']
plt.scatter(x1,y1)
plt.xlabel('Tv advertising amount(in thousands)')
plt.ylabel('Sales amount(in lakhs)')
plt.title('Advertising[TV] and Sales')

#Task 4(Part 2)
fig = plt.figure(figsize = (20,5))
ax3 = plt.subplot(131)
x1 = df_marketing['TV'].values.reshape(-1,1)
y1 = df_marketing['sales']
plt.scatter(x1,y1,color = 'seagreen')
plt.xlabel('Tv advertising amount(in thousands)')
plt.ylabel('Sales amount(in lakhs)')
plt.title('Advertising[TV] and Sales')

ax3 = plt.subplot(132)
x2 = df_marketing['newspaper'].values.reshape(-1,1)
y2 = df_marketing['sales']
plt.scatter(x2,y2,color = 'darkgreen')
plt.xlabel('Newspaper advertising amount(in thousands)')
plt.ylabel('Sales amount(in lakhs)')
plt.title('Advertising[Newspaper] and Sales')

ax3 = plt.subplot(133)
x3 = df_marketing['radio'].values.reshape(-1,1)
y3 = df_marketing['sales']
plt.scatter(x3,y3,color = 'black')
plt.xlabel('Radio advertising amount(in thousands)')
plt.ylabel('Sales amount(in lakhs)')
plt.title('Advertising[Radio] and Sales')

#Task 4(Part 3)
sns.pairplot(df_marketing,vars = ['TV','radio','newspaper'],hue = 'sales')

#Task 5(Part 1)
df_marketing['TotalAdvt'] = sum([df_marketing['TV'],df_marketing['radio'],df_marketing['newspaper']])

#Task 5(Part 2)
x = df_marketing['TotalAdvt'].values.reshape(-1,1)
y = df_marketing['sales']

simple_model = linear_model.LinearRegression()

simple_model.fit(x,y)

simple_model.intercept_

simple_model.coef_

simple_model.score(x,y)

#Task 5(Part 3)
#4.9782741*x +412.94534013284647 

#Task 5(Part 4)
simple_model.predict(50000)

#Task 6(Part 1)
xm = df_marketing.iloc[:,1:4].values
ym = df_marketing['sales']

multi_model = linear_model.LinearRegression()

multi_model.fit(xm,ym)

multi_model.coef_

multi_model.intercept_

multi_model.score(xm,ym)

#Task 6(Part 2)
# 4.68636807*x1+18.76042293*x2+0.14346244*x3+283.19952964638355

# 4.68636807*90000+18.76042293*3000+0.14346244*45000+283.19952964638355
multi_model.predict(np.array([90000,3000,45000]).reshape(1,-1))

# 4.68636807*290000+18.76042293*0+0.14346244*80000+283.19952964638355
multi_model.predict(np.array([290000,0,80000]).reshape(1,-1))


#Task 7
X = sm.add_constant(x)
model_ols = sm.OLS(y,X)
result_ols = model_ols.fit()

print(result_ols.summary())


X = sm.add_constant(xm)
model_ols = sm.OLS(ym,X)
result_ols = model_ols.fit()

print(result_ols.summary())


simple_y = simple_model.predict(x)

multi_y = multi_model.predict(xm)



simple_mse = metrics.mean_squared_error(y,simple_y)

multi_mse = metrics.mean_squared_error(ym,multi_y)



