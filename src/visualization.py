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