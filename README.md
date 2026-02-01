# Heart Disease Classification – ML Assignment 2

## 1. Problem Statement
Cardiovascular diseases are among the leading causes of mortality worldwide. Early prediction of heart disease can help in timely diagnosis and preventive healthcare.  
The objective of this project is to implement and evaluate multiple machine learning classification models to predict the presence of heart disease based on patient clinical attributes.

---

## 2. Dataset Description
- **Dataset Source:** UCI Heart Disease Dataset (Extended Version)
- **Problem Type:** Binary Classification
- **Target Variable (`num`):**
  - `0` → No Heart Disease
  - `> 0` → Presence of Heart Disease
- **Number of Records:** More than 500
- **Number of Features:** 13+

### Features
- age
- sex
- dataset
- cp (chest pain type)
- trestbps (resting blood pressure)
- chol (serum cholesterol)
- fbs (fasting blood sugar)
- restecg (resting ECG)
- thalch (maximum heart rate achieved)
- exang (exercise induced angina)
- oldpeak (ST depression)
- slope
- ca (number of major vessels)
- thal

### Data Preprocessing
- Missing values handled using:
  - Median imputation for numerical features
  - Most frequent imputation for categorical features
- Categorical features encoded using One-Hot Encoding
- Numerical features scaled using StandardScaler
- ID column removed
- Target variable converted to binary

---

## 3. Models Used and Evaluation Metrics
The following machine learning models were implemented using the same dataset and train–test split.  
Evaluation was performed using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|---------|-----|----------|--------|---------|-----|
| Logistic Regression | 0.8424 | 0.9256 | 0.8349 | 0.8922 | 0.8626 | 0.6804 |
| Decision Tree | 0.8152 | 0.8422 | 0.8148 | 0.8627 | 0.8381 | 0.6247 |
| K-Nearest Neighbors | 0.8478 | 0.9049 | 0.8426 | 0.8922 | 0.8667 | 0.6913 |
| Naive Bayes | 0.8478 | 0.9072 | 0.8558 | 0.8725 | 0.8641 | 0.6914 |
| Random Forest (Ensemble) | **0.8587** | **0.9310** | **0.8585** | **0.8922** | **0.8750** | **0.7133** |
| XGBoost (Ensemble) | 0.8533 | 0.9139 | 0.8505 | 0.8922 | 0.8708 | 0.7023 |

---

## 4. Observations on Model Performance

| ML Model | Observation |
|--------|------------|
| Logistic Regression | Achieved strong performance with high AUC, indicating good class separability, but limited in capturing non-linear relationships. |
| Decision Tree | Showed comparatively lower performance due to overfitting tendencies despite depth control. |
| K-Nearest Neighbors | Delivered good recall and balanced performance, highly dependent on feature scaling and choice of K. |
| Naive Bayes | Provided stable results with fast training time, though based on strong feature independence assumptions. |
| Random Forest (Ensemble) | Performed the best overall with the highest accuracy, AUC, and MCC due to ensemble learning and reduced variance. |
| XGBoost (Ensemble) | Demonstrated strong predictive capability with an effective bias–variance tradeoff, slightly lower than Random Forest on this dataset. |
