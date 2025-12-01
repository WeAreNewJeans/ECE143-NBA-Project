import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import os
from dataloader import NBAGameDataModule
import pickle


def train_logistic_baseline(datamodule):
    """
    Train a simple logistic regression baseline model.
    Concatenates team1 and team2 features (480-dim) and trains logistic regression.

    Args:
        datamodule: NBAGameDataModule instance
    """
    print("="*80)
    print("Logistic Regression Baseline")
    print("="*80)

    # Setup data
    print("\nSetting up data...")
    datamodule.setup()

    # Get dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # Collect training data
    print("\nCollecting training data...")
    X_train = []
    y_train = []

    for team1_features, team2_features, labels in train_loader:
        # Concatenate team features: [team1(240), team2(240)] = 480-dim
        batch_features = torch.cat([team1_features, team2_features], dim=1)
        X_train.append(batch_features.numpy())
        y_train.append(labels.squeeze().numpy())

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    # Collect validation data
    print("\nCollecting validation data...")
    X_val = []
    y_val = []

    for team1_features, team2_features, labels in val_loader:
        batch_features = torch.cat([team1_features, team2_features], dim=1)
        X_val.append(batch_features.numpy())
        y_val.append(labels.squeeze().numpy())

    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)

    print(f"Validation data shape: {X_val.shape}")

    # Collect test data
    print("\nCollecting test data...")
    X_test = []
    y_test = []

    for team1_features, team2_features, labels in test_loader:
        batch_features = torch.cat([team1_features, team2_features], dim=1)
        X_test.append(batch_features.numpy())
        y_test.append(labels.squeeze().numpy())

    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    print(f"Test data shape: {X_test.shape}")

    # Train logistic regression
    print("\n" + "="*80)
    print("Training Logistic Regression...")
    print("="*80)

    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        verbose=1,
        n_jobs=-1,  # Use all CPU cores
        C=1.0,  # Regularization strength
        solver='lbfgs'
    )

    lr_model.fit(X_train, y_train)

    print("\nTraining complete!")

    # Evaluate on training set
    print("\n" + "="*80)
    print("Training Set Performance")
    print("="*80)

    y_train_pred = lr_model.predict(X_train)
    y_train_proba = lr_model.predict_proba(X_train)[:, 1]

    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Training AUC-ROC: {train_auc:.4f}")

    # Evaluate on validation set
    print("\n" + "="*80)
    print("Validation Set Performance")
    print("="*80)

    y_val_pred = lr_model.predict(X_val)
    y_val_proba = lr_model.predict_proba(X_val)[:, 1]

    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation AUC-ROC: {val_auc:.4f}")

    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Loss', 'Win']))

    # Evaluate on test set
    print("\n" + "="*80)
    print("Test Set Performance")
    print("="*80)

    y_test_pred = lr_model.predict(X_test)
    y_test_proba = lr_model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC-ROC: {test_auc:.4f}")

    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Loss', 'Win']))

    # Save model
    model_path = 'baseline_logistic_regression.pkl'
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(lr_model, f)

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Training Accuracy:   {train_acc:.4f}  |  AUC: {train_auc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}  |  AUC: {val_auc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}  |  AUC: {test_auc:.4f}")

    # Analyze feature importance (top coefficients)
    print("\n" + "="*80)
    print("Top 20 Most Important Features (by absolute coefficient)")
    print("="*80)

    coefficients = lr_model.coef_[0]
    feature_importance = np.abs(coefficients)
    top_indices = np.argsort(feature_importance)[-20:][::-1]

    # Feature names (simplified)
    stat_names = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TOV', 'MPG', 'FG%', '3P%', 'FT%']

    for idx in top_indices:
        team = "Team1" if idx < 240 else "Team2"
        player_idx = (idx % 240) // 20
        stat_idx = (idx % 240) % 20

        if stat_idx < 10:
            stat_type = "Season"
            stat_name = stat_names[stat_idx]
        else:
            stat_type = "Recent"
            stat_name = stat_names[stat_idx - 10]

        coef = coefficients[idx]
        print(f"Feature {idx:3d}: {team} Player{player_idx:2d} {stat_type:6s} {stat_name:4s} = {coef:+.4f}")

    return lr_model, {
        'train_acc': train_acc,
        'train_auc': train_auc,
        'val_acc': val_acc,
        'val_auc': val_auc,
        'test_acc': test_acc,
        'test_auc': test_auc
    }


if __name__ == "__main__":
    # Pre-computed features directory
    precomputed_dir = r"c:\Users\jeffc\Desktop\ECE 143\project\data\precomputed_features"

    # Check if pre-computed features exist
    if not os.path.exists(precomputed_dir):
        print("="*80)
        print("ERROR: Pre-computed features not found!")
        print("="*80)
        print(f"\nDirectory not found: {precomputed_dir}")
        print("\nPlease run the following command first:")
        print("  python precompute_features.py")
        print("\nThis will pre-compute all features and save them to disk.")
        exit(1)

    # Initialize DataModule with pre-computed features
    print("Initializing DataModule with pre-computed features...")
    datamodule = NBAGameDataModule(
        precomputed_dir=precomputed_dir,
        batch_size=256,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    # Train baseline
    lr_model, metrics = train_logistic_baseline(datamodule)

    print("\n" + "="*80)
    print("Baseline Training Complete!")
    print("="*80)
