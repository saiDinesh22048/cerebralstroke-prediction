import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  StandardScaler
from imblearn.combine import SMOTEENN

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
  
with st.sidebar:
  st.header('Input features')
  gender = st.selectbox('Gender', ('Male', 'Female'))
  age = st.slider('Age', 25, 60, 43)
  hypo = st.radio("hypertension", ("Yes", "No"))
  hypo = 1 if hypo == "Yes" else 0
  heart= st.radio("heart disease", ("Yes", "No"))
  heart = 1 if heart == "Yes" else 0
  marry_status= st.radio("Marrital Status", ("Yes", "No"))
  wrk_typ = st.selectbox('Work type', ('Private', 'Self-employed','Never Worked','children','Govt_job'))
  gls_lvl = st.slider('Glucose level', 0, 600, 85)
  bmi = st.slider('Bmi', 5, 100, 30)
  smking_stat= st.selectbox('Smoking status', ('never smoked', 'formerly smoked','smokes'))
  # Create a DataFrame for the input features
  label_encoder = LabelEncoder()
  data = {'gender': gender,
          'age': age,
          'hypertension': hypo,
          'heart_disease': heart,
          'ever_married': marry_status,
          'work_type': wrk_typ,
          'avg_glucose_level': gls_lvl,
          'bmi': bmi,
          'smoking_status': smking_stat}


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

st.write(input_values.shape)
st.write(y_raw.shape)

smote_enn = SMOTEENN()
X_res1, y_res1 = smote_enn.fit_resample(input_values ,y_raw)
input = input.reshape(1, -1)
input
input_values

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
st.write(y)
