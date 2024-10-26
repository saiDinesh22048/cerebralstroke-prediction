import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  StandardScaler
from imblearn.combine import SMOTEENN

import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Union

app = FastAPI()

# Define your input data schema for validation
class PredictionInput(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

st.title('Cerebral Stroke Predictor')

st.info('This app predicts the cerebral stroke using a machine learning algorithm!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('dataset.csv')
  df


  st.write('**X**')
  df = df.drop('id', axis=1)
  df = df.drop('Residence_type', axis=1)
  df = df[(df['age'] >= 25) & (df['bmi'] <= 60)]
  df['bmi'].fillna(df['bmi'].mean(), inplace=True)
  X_raw= df.drop('stroke', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.stroke
  y_raw

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='hypertension', y='age', color='stroke')
  
@app.post("/predict/")
def predict(input_data: PredictionInput):
    # Create a DataFrame for the input features
    label_encoder = LabelEncoder()
    data = {'gender': input_data.gender,
            'age': input_data.age,
            'hypertension':input_data.hypertension,
            'heart_disease': input_data.heart_disease,
            'ever_married': input_data.ever_married,
            'work_type': input_data.work_type,
            'avg_glucose_level': input_data.avg_glucose_level,
            'bmi': input_data.bmi,
            'smoking_status':input_data.smoking_status}
    
    
    input_df = pd.DataFrame(data, index=[0])
    input_values = pd.concat([input_df, X_raw], axis=0)
    
    input_values['gender'] = label_encoder.fit_transform(input_values['gender'])
    
    input_values['hypertension'] = label_encoder.fit_transform(input_values['hypertension'])
    
    input_values['heart_disease'] = label_encoder.fit_transform(input_values['heart_disease'])
    
    input_values['ever_married'] = label_encoder.fit_transform(input_values['ever_married'])
    
    input_values['smoking_status'] = input_values['smoking_status'].fillna('Unknown')
    
    input_values['smoking_status'] = label_encoder.fit_transform(input_values['smoking_status'])
    
    input_values['work_type'] = label_encoder.fit_transform(input_values['work_type'])
    
    
    scaler =StandardScaler()
    input_values = scaler.fit_transform(input_values)
    
    arr = np.array(input_values)
    input =arr[0, :]
    input_values =arr[1:, :]
    
    with st.expander('Input features'):
      st.write('**Input values**')
      input_df
      st.write('**Combined input data**')
      input_values
    
    smote_enn = SMOTEENN()
    X_res1, y_res1 = smote_enn.fit_resample(input_values ,y_raw)
    input = input.reshape(1, -1)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_curve,auc
    
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_res1, y_res1, test_size=0.2, random_state=42)
    
    # Initializing the Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    
    # Training the classifier
    rf_classifier.fit(X_train, y_train)
    
    # Making predictions
    y = rf_classifier.predict(input)
    if y[0]==0:
      return "congrats,you have less risk for cerebral stroke"
    else:
      return "you have high risk for cerebral stroke.please consult a neurologist immideatly"
