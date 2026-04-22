import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df=pd.read_csv("C:\\Users\ACER\Downloads\\boston_data.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df.head()
df.info()
df.describe()
df.isnull().sum()
plt.figure(figsize=(42,42))
sns.pairplot(df)
sns.displot(df['medv'])
sns.heatmap(df.corr(),annot=True)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
X=df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax','ptratio', 'black', 'lstat']]
y=df['medv']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#linearregressor
model_for_pred=LinearRegression()
scalemode=StandardScaler()

x_train_scaled=scalemode.fit_transform(x_train)
x_test=scalemode.transform(x_test)

model_for_pred.fit(x_train_scaled, y_train)
y_pred=model_for_pred.predict(x_test)
error=mean_squared_error(y_test,y_pred)
error2=mean_absolute_error(y_test,y_pred)
r2_1=r2_score(y_test,y_pred)

#Ridge
ridge=Ridge(alpha=1.0)
ridge.fit(x_train_scaled,y_train)
y_pred2=ridge.predict(x_test)
error_of_ridge=mean_squared_error(y_test,y_pred2)
error_of_ridge2=mean_absolute_error(y_test,y_pred2)
r2_2=r2_score(y_test,y_pred2)

#randomforest
forest=RandomForestRegressor(random_state=42)
forest.fit(x_train_scaled,y_train)
y_pred3=forest.predict(x_test)
error_of_forest=mean_squared_error(y_test,y_pred3)
error_of_forest2=mean_absolute_error(y_test,y_pred3)
r2_3=r2_score(y_test,y_pred3)

#comparison of models
results = pd.DataFrame({'LinearRegressor':[error,error2,r2_1],
                        'Ridge':[error_of_ridge,error_of_ridge2,r2_2],
                        'Forest':[error_of_forest,error_of_forest2,r2_3],},
                       index=['mse','mae','r2'])
plt.figure()
plt.scatter(y_test,y_pred)
plt.title('Linear Regression')

plt.figure()
plt.scatter(y_test,y_pred2)
plt.title('Ridge')

plt.figure()
plt.scatter(y_test,y_pred3)
plt.title('RandomForest')

print(results)
print('the best model:' + str(results.loc['r2'].idxmax()))