## Credit Card Fraud Detection: Unsupervised & Supervised Learning Approaches

This project implements two robust pipelines to detect fraudulent credit card transactions:

1. **Unsupervised Learning** using an Autoencoder neural network.
2. **Supervised Learning** using classification models with SMOTE for class imbalance.

The Autoencoder is trained only on legitimate transactions and detects fraud based on reconstruction error. The supervised models use labeled data with SMOTE to improve fraud detection on a highly imbalanced dataset.

### Technologies Used

- Python, NumPy, Pandas
- Matplotlib, Seaborn (visualization)
- Scikit-learn (metrics, preprocessing)
- TensorFlow/Keras (autoencoder)
- XGBoost, LightGBM, CatBoost, Random Forest (supervised models)
- EarlyStopping and ModelCheckpoint (training optimization)

### Dataset

- Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions total
- 492 fraudulent cases (~0.17%)

### Autoencoder-Based Anomaly Detection

- **Approach**: Trained only on normal transactions to learn a low-dimensional representation.
- **Detection**: Reconstruction error is used to flag anomalies.
- **Thresholding**: Optimized based on F1 score using validation data.

**Evaluation Metrics:**
- Precision: 0.8581
- Recall: 0.7652
- F1 Score: 0.8089
- ROC AUC: 0.9496

### Supervised Classification (with SMOTE)

- **Approach**: Used Random Forest, XGBoost, LightGBM, CatBoost
- **Oversampling**: SMOTE used to balance minority class
- **Train/Test Split**: Stratified split to preserve fraud ratio

**Best Model Results (e.g., XGBoost):**
- Accuracy: 0.9994
- Precision: 0.9877
- Recall: 0.9795
- F1 Score: 0.9836
- ROC AUC: 0.9998

### Summary

This project showcases a complete and comparative approach to fraud detection using both unsupervised and supervised machine learning techniques. It balances detection accuracy with model generalization and interpretability, making it suitable for real-world deployment.
