import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
import os
from dataloaders.dataloader import NBAGameDataModule
import pickle
import matplotlib.pyplot as plt
import matplotlib
import warnings
from sklearn.exceptions import ConvergenceWarning

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def train_logistic_baseline(datamodule):
    """
    Train a simple logistic regression baseline model.
    Concatenates team1 and team2 features (480-dim) and trains logistic regression.

    Args:
        datamodule: NBAGameDataModule instance
    """

    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # Collect training data
    X_train = []
    y_train = []

    for team1_features, team2_features, labels in train_loader:
        batch_features = torch.cat([team1_features, team2_features], dim=1)
        X_train.append(batch_features.numpy())
        y_train.append(labels.squeeze().numpy())

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    # Collect validation data
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
    X_test = []
    y_test = []

    for team1_features, team2_features, labels in test_loader:
        batch_features = torch.cat([team1_features, team2_features], dim=1)
        X_test.append(batch_features.numpy())
        y_test.append(labels.squeeze().numpy())

    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    print(f"Test data shape: {X_test.shape}")

    history = {
        "iterations": [],
        "train_loss": [],
        "train_acc": [],
        "train_auc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
    }

    lr_model = LogisticRegression(
        max_iter=1,
        random_state=42,
        verbose=0,
        n_jobs=1,
        C=1.0,
        solver="lbfgs",
        warm_start=True
    )

    total_iterations = 100
    check_points = list(range(1, total_iterations + 1, 1))  # Record every n iteration

    print(f"Training with {total_iterations} iterations.")

    for iteration in range(1, total_iterations + 1):
        lr_model.max_iter = iteration
        lr_model.fit(X_train, y_train)

        if iteration in check_points or iteration == 1:
            # Training set
            y_train_pred = lr_model.predict(X_train)
            y_train_proba = lr_model.predict_proba(X_train)
            train_loss = log_loss(y_train, y_train_proba)
            train_acc = accuracy_score(y_train, y_train_pred)
            train_auc = roc_auc_score(y_train, y_train_proba[:, 1])

            # Validation set
            y_val_pred = lr_model.predict(X_val)
            y_val_proba = lr_model.predict_proba(X_val)
            val_loss = log_loss(y_val, y_val_proba)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_auc = roc_auc_score(y_val, y_val_proba[:, 1])

            # Record metrics
            history["iterations"].append(iteration)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_auc"].append(train_auc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_auc"].append(val_auc)

            print(
                f"Iteration {iteration:3d} - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}"
            )

    print("\nTraining complete!")

    # Evaluate on training set
    print("\n" + "=" * 80)
    print("Training Set Performance")

    y_train_pred = lr_model.predict(X_train)
    y_train_proba = lr_model.predict_proba(X_train)[:, 1]

    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Training AUC-ROC: {train_auc:.4f}")

    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("Validation Set Performance")

    y_val_pred = lr_model.predict(X_val)
    y_val_proba = lr_model.predict_proba(X_val)[:, 1]

    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation AUC-ROC: {val_auc:.4f}")

    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=["Loss", "Win"]))

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Test Set Performance")

    y_test_pred = lr_model.predict(X_test)
    y_test_proba = lr_model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC-ROC: {test_auc:.4f}")

    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Loss", "Win"]))

    model_path = "baseline_logistic_regression.pkl"
    print(f"\nSaving model to {model_path}.")
    with open(model_path, "wb") as f:
        pickle.dump(lr_model, f)

    print(f"Training Accuracy:   {train_acc:.4f}  |  AUC: {train_auc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}  |  AUC: {val_auc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}  |  AUC: {test_auc:.4f}")

    print("\nTop 20 Most Important Features (by absolute coefficient)")

    coefficients = lr_model.coef_[0]
    feature_importance = np.abs(coefficients)
    top_indices = np.argsort(feature_importance)[-20:][::-1]

    # Feature names
    stat_names = ["PPG", "RPG", "APG", "SPG", "BPG", "TOV", "MPG", "FG%", "3P%", "FT%"]

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

    print("\n" + "=" * 80)
    print("Generating Training Visualizations...")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    axes[0].plot(history["iterations"], history["train_loss"], "b-", label="Training", linewidth=2)
    axes[0].plot(history["iterations"], history["val_loss"], "r-", label="Validation", linewidth=2)
    axes[0].set_xlabel("Iterations", fontsize=12)
    axes[0].set_ylabel("Loss (Log Loss)", fontsize=12)
    axes[0].set_title("Training Progress - Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(history["iterations"], history["train_acc"], "b-", label="Training", linewidth=2)
    axes[1].plot(history["iterations"], history["val_acc"], "r-", label="Validation", linewidth=2)
    axes[1].set_xlabel("Iterations", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training Progress - Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.5, 1.0])

    # AUC curve
    axes[2].plot(history["iterations"], history["train_auc"], "b-", label="Training", linewidth=2)
    axes[2].plot(history["iterations"], history["val_auc"], "r-", label="Validation", linewidth=2)
    axes[2].set_xlabel("Iterations", fontsize=12)
    axes[2].set_ylabel("AUC-ROC", fontsize=12)
    axes[2].set_title("Training Progress - AUC", fontsize=14, fontweight="bold")
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0.5, 1.0])

    plt.tight_layout()

    plot_path = "baseline_logistic_training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nTraining curves saved to: {plot_path}")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    metrics = ["Accuracy", "AUC-ROC"]
    train_scores = [train_acc, train_auc]
    val_scores = [val_acc, val_auc]
    test_scores = [test_acc, test_auc]

    x = np.arange(len(metrics))
    width = 0.25

    bars1 = ax.bar(x - width, train_scores, width, label="Training", color="#2E86AB")
    bars2 = ax.bar(x, val_scores, width, label="Validation", color="#A23B72")
    bars3 = ax.bar(x + width, test_scores, width, label="Test", color="#F18F01")

    ax.set_xlabel("Evaluation Metrics", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Logistic Regression Baseline - Final Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0.5, 1.0])

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    plt.tight_layout()

    # Save plot
    comparison_path = "baseline_logistic_performance_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    print(f"Performance comparison plot saved to: {comparison_path}")
    plt.close()

    # Save training history data
    history_df = pd.DataFrame(history)
    history_path = "baseline_logistic_training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Training history data saved to: {history_path}")

    return lr_model, {
        "train_acc": train_acc,
        "train_auc": train_auc,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "history": history,
    }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    precomputed_dir = os.path.join(script_dir, "data", "precomputed_features")

    if not os.path.exists(precomputed_dir):
        print("=" * 80)
        print("ERROR: Pre-computed features not found!")
        print("=" * 80)
        print(f"\nDirectory not found: {precomputed_dir}")
        print("\nPlease run recompute_features.py first.")
        exit(1)

    print(f"Precomputed directory: {precomputed_dir}")
    datamodule = NBAGameDataModule(
        precomputed_dir=precomputed_dir, batch_size=256, num_workers=4
    )

    lr_model, metrics = train_logistic_baseline(datamodule)

    print("Baseline Training Complete!")
