import pandas
import numpy as np
data = {'distance':np.random.uniform(1,50,1000),
        'weight':np.random.uniform(1,50,1000),
        'average_speed':np.random.uniform(1,50,1000),
    'vehicle_type':np.random.choice(['bicycle','bike','car'],size=1000)}
speed_factor = {'bicycle': 0.4, 'bike': 0.7, 'car': 1.5}
vehicle_delay = {'bicycle': 50.0, 'bike': 20.0, 'car': 2.0}

data['adjusted_speed'] = [
    s * speed_factor[v] for s, v in zip(data['average_speed'], data['vehicle_type'])
]

data['delivery_time'] = (
    data['distance'] / data['adjusted_speed']
    + np.random.normal(0, 0.3, 1000)
    + [vehicle_delay[v] for v in data['vehicle_type']]
)
data['delivery_time'] += data['weight'] * 0.02
df = pandas.DataFrame(data)
print(df.head())

#%%
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data=df['delivery_time'],kde=True)
plt.title('delivery time')
plt.xlabel('time')
plt.ylabel('quantity')
plt.show()
sns.scatterplot(x='distance',y='delivery_time',hue='vehicle_type',data=df)
plt.title('how distance influences delivery time')
plt.xlabel('distance')
plt.ylabel('delivery_time')
plt.show()
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df['vehicle_type'] = LabelEncoder().fit_transform(df['vehicle_type'])
x=df[['distance','weight','vehicle_type']]
y=df['delivery_time']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
model = RandomForestRegressor(random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print('mean squared error: ',mse)
print('mean absolute error: ',mae)
importance = pd.Series(model.feature_importances_, index=x.columns)
print(importance.sort_values(ascending=False))
#%%
import joblib
joblib.dump(model, 'model_for_predicting_time_of_delivery.pkl')