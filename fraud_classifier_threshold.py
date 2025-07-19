import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from catboost import CatBoostClassifier
import joblib

# Load dataset
df = pd.read_csv("C:/Users/Manoj/Desktop/dataset_converted.csv")

# Identify categorical and numerical columns
cat_cols = [col for col in df.columns if col.startswith("cat")]
num_cols = [col for col in df.columns if col.startswith("cont")]

# Create binary target (fraud = top 5% losses)
df['is_fraud'] = (df['loss'] > df['loss'].quantile(0.95)).astype(int)

# Select top 3 numerical features by correlation
correlations = df[num_cols + ['is_fraud']].corr()['is_fraud'].abs().sort_values(ascending=False)
top_nums = correlations.index[1:4].tolist()

# Select top 2 categorical features by Chi-squared approximation
cat_impact = {}
for col in cat_cols:
    table = pd.crosstab(df[col], df['is_fraud'])
    if table.shape[0] > 1:
        chi2_score = ((table - table.mean())**2 / table.mean()).sum().sum()
        cat_impact[col] = chi2_score
top_cats = sorted(cat_impact, key=cat_impact.get, reverse=True)[:2]

# Prepare features and target
features = top_cats + top_nums
X = df[features].copy()
y = df['is_fraud']

# Convert categorical features
for col in top_cats:
    X[col] = X[col].astype("category")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Model training with class_weights
print("🚀 Training CatBoostClassifier...")
model = CatBoostClassifier(verbose=100, random_state=42, class_weights=[1, 10])
model.fit(X_train, y_train, cat_features=top_cats)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Evaluate at custom thresholds
thresholds = [0.9, 0.8, 0.7, 0.6]
for thresh in thresholds:
    print(f"\n🔍 Evaluation at Threshold: {thresh}")
    y_pred = (y_probs > thresh).astype(int)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", round(roc_auc_score(y_test, y_probs), 4))

# Save the model
joblib.dump(model, "C:/Users/Manoj/Desktop/MS/eda_i/fraud_classifier_catboost_tuned.joblib")
print("\n✅ Model saved as fraud_classifier_catboost_tuned.joblib")
print(f"Top categorical: {top_cats} | Top numerical: {top_nums}")
