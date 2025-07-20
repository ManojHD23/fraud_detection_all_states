# fraud_detection_all_states

# 🚨 Insurance Fraud Detection using CatBoost

A full-stack machine learning pipeline to detect fraud in insurance policies using top influencing features and custom decision thresholds. Built with business impact in mind.

## 💼 Business Overview

- 📊 188,318 policies analyzed
- 💰 ₹571.9M total loss exposure
- 🎯 Focus: Top 5% fraudulent claims (high-loss)
- 📉 Reduced false negatives using threshold-tuned CatBoostClassifier
- 📈 Model precision improved for high-risk detection

## ✅ Features

- Top 3 numerical + Top 2 categorical features for model input
- Custom threshold tuning for fraud prediction
- SHAP explainability and visualizations
- Streamlit dashboard for stakeholder usage

## 🧠 Model Performance

- ROC AUC: `0.72`
- Threshold-adjusted recall for fraud: up to `0.13`
- Business segmentation (High, Medium, Low Risk)
- Potential cost savings: ₹60M+ (annual estimate)

## 🚀 How to Use

```bash
# Train the model
python fraud_classifier_threshold.py

# Run the dashboard
streamlit run fraud_prediction_app.py
