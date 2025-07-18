def plot_roc_curves(models, X_test_scaled, X_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(10, 7))

    for name, model in models.items():
        if name == "Logistic Regression":
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            probs = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()


def shap_summary_plot(model, X_test):
    import shap
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:100])
    shap.summary_plot(shap_values, X_test[:100]) 

def plot_class_distribution(y_before, y_after):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.countplot(x=y_before, ax=axes[0])
    axes[0].set_title('Class Distribution Before SMOTE')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')

    sns.countplot(x=y_after, ax=axes[1])
    axes[1].set_title('Class Distribution After SMOTE')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(data):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(14, 10))
    sns.heatmap(data.corr(), cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()