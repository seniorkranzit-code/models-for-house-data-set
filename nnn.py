import numpy as np
import pandas as pd
import joblib
import streamlit as st
model = joblib.load('model_for_predicting_time.pkl')
st.title('delivery time prediction')
st.header('download data')
uploaded_file = st.file_uploader('download data csv',type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('downloading data')
    st.dataframe(df)
    required_columns = ['distance','weight','average speed','vehicle type']
    if all(col in df.columns for col in required_columns):
        vehicle_type_map={'car':0,'truck':1,'bike':2}
        df['vehicle type'] = df['vehicle type'].map(vehicle_type_map)
        predictions=model.predict(df[required_columns])
        df['predictions'] = predictions
        st.write('predicted time of delivery')
        st.dataframe(df)
    else:
        st.error(f'Please select a column {required_columns}')
st.header('manually choose data')
distance=st.slider('distance',1,50,10)
vehicle_type=st.selectbox('vehicle type',['car','truck','bike'])
average_speed=st.slider('average speed',10,80,40)
weight=st.slider('weight',1,100,10)
vehicle_type_map={'car':0,'truck':1,'bike':2}
vehicle_type_encoded=vehicle_type_map[vehicle_type]

if st.button('delivery time'):
    input_data=np.array([[distance,weight,average_speed,vehicle_type_encoded]])
    prediction=model.predict(input_data)
    st.write(f'prediction: {prediction[0]:.2f} minutes')