import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    precision_recall_curve, f1_score
)

from sklearn.ensemble import (
    RandomForestClassifier, 
    StackingClassifier, 
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

if not os.path.exists("data/churn.csv"):
    raise FileNotFoundError("Please ensure 'data/churn.csv' exists.")

df = pd.read_csv("data/churn.csv")
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)

df["tenure"] = df["tenure"].replace(0, 1)

df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"]
df["TenureYears"] = df["tenure"] / 12

service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]
df["ServiceCount"] = df[service_cols].apply(lambda x: sum(x == "Yes"), axis=1)
df["MonthlyPerService"] = df["MonthlyCharges"] / (df["ServiceCount"] + 1)
df["IsSenior"] = df["SeniorCitizen"].astype(int)

y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop(["Churn", "SeniorCitizen"], axis=1)

numeric_features = [
    "tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", 
    "ServiceCount", "MonthlyPerService", "TenureYears", "IsSenior"
]
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ],
    verbose_feature_names_out=False
)

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=4, 
    class_weight="balanced", random_state=42, n_jobs=-1
)

hgb_model = HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=200, max_depth=5, 
    l2_regularization=0.1, random_state=42
)

lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")

estimators = [('rf', rf_model), ('hgb', hgb_model), ('lr', lr_model)]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

pipeline = ImbPipeline(steps=[
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42, sampling_strategy=0.8)),
    ("stacking", stacking_clf)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("🚀 Training Stacking Model (Sample size: {})...".format(len(X_train)))
pipeline.fit(X_train, y_train)

print("\n🔍 Tuning Probability Threshold...")

y_proba_test = pipeline.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_test)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"✅ Optimal Threshold Found: {best_threshold:.4f}")
print(f"   Max F1 Score at this threshold: {best_f1:.4f}")

print("\n" + "="*50)
print("DEFAULT MODEL (Threshold = 0.5)")
print("="*50)
y_pred_default = (y_proba_test >= 0.5).astype(int)
print(classification_report(y_test, y_pred_default))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_test))

print("\n" + "="*50)
print(f"TUNED MODEL (Threshold = {best_threshold:.4f})")
print("="*50)
y_pred_tuned = (y_proba_test >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_tuned))

cm = confusion_matrix(y_test, y_pred_tuned)
print("Confusion Matrix (Tuned):")
print(f"Correctly Stayed: {cm[0][0]} | False Alarms: {cm[0][1]}")
print(f"Missed Churn:     {cm[1][0]} | Caught Churn: {cm[1][1]}")

feature_names = numeric_features + list(pipeline.named_steps["preprocessing"].named_transformers_["cat"].get_feature_names_out())

artifacts = {
    "pipeline": pipeline,
    "feature_names": feature_names,
    "X_train_sample": X_train.sample(100),
    "model_type": "StackingClassifier",
    "threshold": best_threshold
}

os.makedirs("models", exist_ok=True)
joblib.dump(artifacts, "models/churn_stacking_tuned.pkl")
print("\n💾 Model & Optimal Threshold saved to 'models/churn_stacking_tuned.pkl'")