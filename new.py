import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
df=pd.read_csv("C:\\Users\ACER\Downloads\\boston_data.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df.head()
info=df.info()
df.describe()
df.isnull().sum()
#plt.figure(figsize=(42,42))
#sns.pairplot(df)
#sns.displot(df['medv'])
#sns.heatmap(df.corr(),annot=True)

X=df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax','ptratio', 'black', 'lstat']]
y=df['medv']
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_for_pred=LinearRegression()
scalemode=StandardScaler()
x_train_scaled=scalemode.fit_transform(x_train)
x_test_scaled=scalemode.transform(x_test)
model_for_pred.fit(x_train_scaled, y_train)

y_pred=model_for_pred.predict(x_test)

plt.scatter(y_test,y_pred)
error=mean_squared_error(y_test,y_pred)
error2=mean_absolute_error(y_test,y_pred)

print('mean sq error:' + str(error),
      '\nmean abs error:'+ str(error2),flush=True)
print('mean sq error of LinearRegressor model:' + str(error),
      '\nmean abs error:'+ str(error2))
print('mean sq error of Ridge model:' + str(error_of_ridge),
      '\nmean abs error of Ridge model:' + str(error_of_ridge2))
print('mean sq error of RandomForest model:' + str(error_of_forest),
      '\nmean abs error of RandomForest model:' + str(error_of_forest2))