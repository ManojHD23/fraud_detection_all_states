import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("fraud_classifier_catboost_tuned.joblib")

# Define input features
cat_features = ['cat109', 'cat116']
num_features = ['cont2', 'cont7', 'cont3']
all_features = cat_features + num_features

st.set_page_config(page_title="🚨 Insurance Fraud Risk Predictor", layout="centered")
st.title("🚨 Insurance Fraud Prediction App")
st.markdown("This app uses a trained **CatBoostClassifier** to predict whether an insurance policy is potentially fraudulent.")

st.header("📥 Enter Policyholder Details")

# Collect user input
user_input = {}
for col in cat_features:
    user_input[col] = st.selectbox(f"Select {col}", options=['A', 'B', 'C', 'D'])  # Replace with known values if available
for col in num_features:
    user_input[col] = st.number_input(f"Enter {col}", min_value=0.0, step=0.01)

threshold = st.slider("🎚️ Set Fraud Threshold", 0.0, 1.0, 0.9, step=0.05)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])
for col in cat_features:
    input_df[col] = input_df[col].astype("category")

if st.button("🚀 Predict Fraud Risk"):
    proba = model.predict_proba(input_df)[0][1]
    prediction = int(proba >= threshold)

    st.subheader("📊 Prediction Result")
    st.markdown(f"**Fraud Probability:** `{proba:.4f}`")
    st.markdown(f"**Prediction (Threshold = {threshold}):** `{['Legitimate', 'Fraud'][prediction]}`")

    if prediction:
        st.error("⚠️ This policy is likely FRAUDULENT")
    else:
        st.success("✅ This policy seems LEGITIMATE")

    # Explanation of business action
    st.markdown("---")
    st.subheader("📌 Business Action Plan")
    if threshold >= 0.9:
        st.write("🔴 **High-Risk Policy** → Block transaction or require further documentation.")
    elif threshold >= 0.8:
        st.write("🟠 **Medium-Risk Policy** → Manual review required.")
    elif threshold >= 0.6:
        st.write("🟡 **Low-Risk Policy** → Flag for future audit.")
    else:
        st.write("🟢 **Safe Policy** → Accept with standard procedures.")

st.markdown("---")
st.caption("App built with 💡 by Manoj for Insurance Fraud Detection")
