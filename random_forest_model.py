from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    val_probs_rf = rf.predict_proba(X_val)[:, 1]
    test_probs_rf = rf.predict_proba(X_test)[:, 1]

    print("\n===== Random Forest =====")
    print("Val Accuracy:", accuracy_score(y_val, val_probs_rf > 0.5))
    print("Val LogLoss:", log_loss(y_val, val_probs_rf))
    print("Val ROC-AUC:", roc_auc_score(y_val, val_probs_rf))
    print("Val Brier:", brier_score_loss(y_val, val_probs_rf))

    print("\nTest Accuracy:", accuracy_score(y_test, test_probs_rf > 0.5))
    print("Test LogLoss:", log_loss(y_test, test_probs_rf))
    print("Test ROC-AUC:", roc_auc_score(y_test, test_probs_rf))
    print("Test Brier:", brier_score_loss(y_test, test_probs_rf))