import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Classification",
    layout="wide"
)

st.title("Heart Disease Classification – ML Assignment 2")
st.write("Upload test data, select a trained model, and view evaluation results.")

# -------------------------------
# Model Selection
# -------------------------------
model_files = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree": "model/Decision_Tree.pkl",
    "KNN": "model/KNN.pkl",
    "Naive Bayes": "model/Naive_Bayes.pkl",
    "Random Forest": "model/Random_Forest.pkl",
    "XGBoost": "model/XGBoost.pkl"
}

selected_model_name = st.selectbox(
    "Select a Model",
    list(model_files.keys())
)

# Load model
model_path = model_files[selected_model_name]
loaded_object = joblib.load(model_path)

# Handle Naive Bayes separately
if selected_model_name == "Naive Bayes":
    preprocessor, model = loaded_object
else:
    model = loaded_object

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    # Drop ID column if present
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # Target Engineering
    if "num" not in df.columns:
        st.error("Target column 'num' not found in uploaded dataset.")
        st.stop()

    df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

    X = df.drop("num", axis=1)
    y = df["num"]

    # -------------------------------
    # Prediction
    # -------------------------------
    if selected_model_name == "Naive Bayes":
        X_transformed = preprocessor.transform(X)
        y_pred = model.predict(X_transformed)
        y_proba = model.predict_proba(X_transformed)[:, 1]
    else:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

    # -------------------------------
    # Metrics
    # -------------------------------
    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
    col1.metric("Precision", f"{precision_score(y, y_pred):.4f}")

    col2.metric("Recall", f"{recall_score(y, y_pred):.4f}")
    col2.metric("F1 Score", f"{f1_score(y, y_pred):.4f}")

    col3.metric("AUC", f"{roc_auc_score(y, y_proba):.4f}")
    col3.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # -------------------------------
    # Classification Report
    # -------------------------------
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=False)
    st.text(report)

else:
    st.info("Please upload a CSV file to proceed.")
