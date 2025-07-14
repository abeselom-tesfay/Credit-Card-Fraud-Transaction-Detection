# 💳 Credit Card Fraud Detection

This project helps detect fraudulent credit card transactions using machine learning. Fraud detection is a serious challenge because fraud cases are very rare and easily missed. To solve this, we used data balancing techniques and tested powerful models to find fraud more accurately.

---

## 🎯 Objective

The goal is to build a reliable model that can **detect fraudulent transactions** from credit card data. We improve performance by:
- Handling class imbalance using **SMOTE**
- Training and comparing several machine learning models
- Using clear metrics and explainable AI (SHAP)

---

## 📊 Dataset

- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraud Cases**: 492 (very rare)
- **Features**: 30 columns including Time, Amount, and anonymized PCA features (V1–V28)
- **Target**: `Class` (0 = normal, 1 = fraud)

---

## 🧪 What's Included?

### ✅ Preprocessing
- Loads data from CSV
- Balances data using **SMOTE**
- Scales features using **StandardScaler**
- Splits data using **stratified train-test split**

### ✅ Models Used
- Logistic Regression
- Random Forest
- XGBoost ✅ *(Best model)*
- LightGBM
- CatBoost

### ✅ Evaluation
- **Precision**, **Recall**, **F1-score**, **ROC-AUC**
- **Cross-validation**
- **Confusion Matrix**
- **ROC Curve** to compare all models

### ✅ Explainability
- **SHAP (SHapley values)** used to interpret predictions
- Visualizes which features impact fraud predictions the most

### ✅ Deployment Ready
- Best model (XGBoost) saved using `joblib`
- Optional: Streamlit app can be built for user interface

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Credit-Card-Fraud-Detection.git

2. Clone this repository:
   ```bash
   pip install -r requirements.txt

3. Open and run the main notebook:
   ```bash
   cd notebooks
   jupyter notebook Credit-Card-Fraud-Detection.ipynb


## 🧠 Key Results
- XGBoost achieved a ROC-AUC of ~0.99
- The model identifies fraud with high precision and recall
- SHAP plots make model decisions interpretable
- Model is saved and ready to deploy