import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss


# logistic regression model
def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    logreg = LogisticRegression(max_iter=5000)
    logreg.fit(X_train, y_train)

    val_probs_lr = logreg.predict_proba(X_val)[:, 1]
    test_probs_lr = logreg.predict_proba(X_test)[:, 1]

    print("\n===== Logistic Regression =====")
    print("Val Accuracy:", accuracy_score(y_val, val_probs_lr > 0.5))
    print("Val LogLoss:", log_loss(y_val, val_probs_lr))
    print("Val ROC-AUC:", roc_auc_score(y_val, val_probs_lr))
    print("Val Brier:", brier_score_loss(y_val, val_probs_lr))

    print("\nTest Accuracy:", accuracy_score(y_test, test_probs_lr > 0.5))
    print("Test LogLoss:", log_loss(y_test, test_probs_lr))
    print("Test ROC-AUC:", roc_auc_score(y_test, test_probs_lr))
    print("Test Brier:", brier_score_loss(y_test, test_probs_lr))