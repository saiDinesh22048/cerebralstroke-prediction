import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('dataset.csv')
  df

  st.write('**X**')
  X_raw = df.drop('stroke', axis=1)
  X_raw = df.drop('id', axis=1)
  X_raw = df.drop('Residence_type', axis=1)
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
  heart= st.radio("heart disease", ("Yes", "No"))
  marry_status= st.radio("Marrital Status", ("Yes", "No"))
  wrk_typ = st.selectbox('Work type', ('Private', 'Self-employed','Never Worked','children','Govt_job'))
  gls_lvl = st.slider('Glucose level', 0, 600, 85)
  bmi = st.slider('Bmi', 5, 100, 30)
  smking_stat= st.selectbox('Smoking status', ('never smoked', 'formerly smoked','smokes'))
  # Create a DataFrame for the input features
  data = {'gender': gender,
          'age': age,
          'hypertension': hypo,
          'heart_disease': heart,
          'ever_married': marry_status,
          'work_type': wrk_typ,
          'avg_glucose_level': gls_lvl,
          'bmi': bmi,
          'smoking_status': smking_stat}

# Custom mapping for 'gender' column
gender_map = {'Male': 1, 'Female': 0}
data['gender'] = data['gender'].map(gender_map)

# Custom mapping for 'ever_married' column
ever_married_map = {'Yes': 1, 'No': 0}
data['ever_married'] = data['ever_married'].map(ever_married_map)

# Special value imputation for 'smoking_status'
data['smoking_status'] = data['smoking_status'].fillna('Unknown')

# Custom mapping for 'smoking_status' column
smoking_status_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}
data['smoking_status'] = data['smoking_status'].map(smoking_status_map)

# Custom mapping for 'work_type' column
work_type_map = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
data['work_type'] = data['work_type'].map(work_type_map)

input_df = pd.DataFrame(data, index=[0])
input_values = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Input values**')
  input_df
  st.write('**Combined input data**')
  input_values



# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                 1: 'Chinstrap',
                                 2: 'Gentoo'})

# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0])) 
"""
