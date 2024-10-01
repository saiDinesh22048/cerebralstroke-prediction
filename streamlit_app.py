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
  df = df.dropna(how='any')
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

st.write(input_values.shape)
st.write(y_raw.shape)

scaler =StandardScaler()
input_values = scaler.fit_transform(input_values)

with st.expander('Input features'):
  st.write('**Input values**')
  input_df
  st.write('**Combined input data**')
  input_values

input =input_values.drop(input_values.index[0])
input_values =input_values.iloc[1:]
smote_enn = SMOTEENN()
X_res1, y_res1 = smote_enn.fit_resample(input_values ,y_raw)

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
