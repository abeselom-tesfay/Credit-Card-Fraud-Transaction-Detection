## Credit Card Fraud Detection using Autoencoder

This project implements an advanced anomaly detection pipeline to detect fraudulent credit card transactions using an **Autoencoder neural network**. The model is trained in an **unsupervised fashion** on legitimate transactions and learns to identify fraud by measuring **reconstruction errors**.

### Technologies Used

- Python, NumPy, Pandas
- Matplotlib, Seaborn (for visualization)
- Scikit-learn (metrics, preprocessing)
- TensorFlow / Keras (deep learning)
- EarlyStopping & ModelCheckpoint for training optimization

### Evaluation Metrics

The model is evaluated on a test set containing both normal and fraudulent samples using:

- **Precision**
- **Recall**
- **F1 Score**
- **ROC AUC**
- **Confusion Matrix**


### Dataset
- The dataset used is the Credit Card Fraud Detection dataset available on Kaggle.
- It contains 284,315 transactions, of which 492 are fraudulent.
