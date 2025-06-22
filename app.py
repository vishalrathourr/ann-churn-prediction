import streamlit as st 
import pickle
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd 
import time 


## Load the trained model, scaler pickle, one hot pickle
model = tf.keras.models.load_model('model.h5')


## load the encoder and scaler
with open('label_enc_gender.pkl', 'rb') as file:
    label_enc_gender = pickle.load(file)

with open('OHE.pkl', 'rb') as file:
    OHE = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔮")

## Streamlit app 
st.title("🔮 Customer Churn Prediction App")
st.write("Predict whether a customer will churn based on their information.")

st.header("👥 Customer Information")

geography = st.selectbox('🌍 Geography', OHE.categories_[0])
gender = st.radio('👤 Gender', label_enc_gender.classes_, horizontal=True)
age = st.slider('🎂 Age', 18, 92, value=30)


st.header("💰 Financial Details")

balance = st.number_input('🏦 Balance ($)', min_value=0.0, value=0.0, step=100.0)
credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=600, step=10)
est_salary = st.number_input('💼 Estimated Salary ($)', min_value=0.0, value=0.0, step=100.0)
tenure = st.slider('📆 Tenure (years)', 0, 10, value=3)
num_of_prods = st.radio('🛒 Number of Products', [1, 2, 3, 4], horizontal=True)


st.header("🏷️ Membership Details")

has_cr_card = st.checkbox('💳 Has Credit Card')
has_cr_card = 1 if has_cr_card else 0

is_active_member = st.checkbox('✅ Is Active Member')
is_active_member = 1 if is_active_member else 0

## Precict button

if st.button('Predict Churn'):

    with st.spinner("Running prediction..."):
        time.sleep(1)  # simulate loading
    ## Prepare input data

        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender':[label_enc_gender.transform([gender])[0]], 
            'Age':[age],
            'Tenure':[tenure], 
            'Balance':[balance], 
            'NumOfProducts': [num_of_prods],
            'HasCrCard':[has_cr_card], 
            'IsActiveMember':[is_active_member], 
            'EstimatedSalary':[est_salary], 
        })

        # OHE 'Geography'
        geo_encoded = OHE.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=OHE.get_feature_names_out(['Geography']))

        ## Combine with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict the churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        # Display probability
        st.subheader("📊 Churn Probability")
        st.progress(int(prediction_proba * 100))

         # Output result
        if prediction_proba > 0.5:
            st.error(f"⚠️ The customer is likely to churn.\nProbability: {prediction_proba:.2%}")
        else:
            st.success(f"✅ The customer is NOT likely to churn.\nProbability: {prediction_proba:.2%}")

