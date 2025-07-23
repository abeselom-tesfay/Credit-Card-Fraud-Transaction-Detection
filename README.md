# 💳 Credit Card Fraud Detection using Autoencoder

This project implements an advanced anomaly detection pipeline to detect fraudulent credit card transactions using an **Autoencoder neural network**. The model is trained in an **unsupervised fashion** on legitimate transactions and learns to identify fraud by measuring **reconstruction errors**.

---

## 🧠 Model Overview

The model uses an **Autoencoder** neural network with the following characteristics:

- Trained only on **normal (non-fraudulent)** transactions.
- Detects fraud based on **high reconstruction error**.
- Threshold is set at the **95th percentile** of normal reconstruction errors.

---

## 🔬 Technologies Used

- Python, NumPy, Pandas
- Matplotlib, Seaborn (for visualization)
- Scikit-learn (metrics, preprocessing)
- TensorFlow / Keras (deep learning)
- EarlyStopping & ModelCheckpoint for training optimization

---

## 📈 Evaluation Metrics

The model is evaluated on a test set containing both normal and fraudulent samples using:

- **Precision**
- **Recall**
- **F1 Score**
- **ROC AUC**
- **Confusion Matrix**

The Autoencoder achieved:

   Precision: 0.1282
   Recall: 0.8557
   F1 Score: 0.2230
   ROC AUC: 0.9459

---

## 📊 Visualizations

- Class distribution before training  
- Reconstruction error histograms  
- Confusion matrix heatmap  
- ROC curve 

## 📌 Notes
- This is an unsupervised learning approach tailored for highly imbalanced fraud detection problems.
- The threshold selection (95th percentile) can be tuned for risk tolerance in real-world applications.

## 📁 Dataset
- The dataset used is the Credit Card Fraud Detection dataset available on Kaggle.
- It contains 284,315 transactions, of which 492 are fraudulent.
