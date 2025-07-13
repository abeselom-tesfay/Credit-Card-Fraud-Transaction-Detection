def get_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_seed=42)
    }

    return models

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"--- {name} ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}\n")

    return {'model': name, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}


def cross_validate_model(model, X, y):
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    return scores.mean(), scores.std()
