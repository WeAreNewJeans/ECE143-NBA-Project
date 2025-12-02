"""
Compare training performance of four models
Including: Logistic Regression, Basic Model, Transformer, Transformer with TeamInfo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# Set font
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# Output to root directory
OUTPUT_DIR = "."


def load_logistic_regression_data():
    """Load Logistic Regression training history"""
    df = pd.read_csv("LR_results/baseline_logistic_training_history.csv")
    # Only take first 77 rows (rest are duplicates)
    df = df[:77]
    return df


def merge_train_val_rows(df):
    """
    Merge training and validation rows by epoch.
    PyTorch Lightning's CSVLogger writes train and val metrics in separate rows.
    """
    # Separate training and validation data
    train_df = df[df["train_acc"].notna()][["epoch", "train_loss", "train_acc"]].copy()
    val_df = df[df["val_acc"].notna()][["epoch", "val_loss", "val_acc"]].copy()

    # Merge by epoch
    merged_df = pd.merge(train_df, val_df, on="epoch", how="outer")
    merged_df = merged_df.sort_values("epoch").reset_index(drop=True)

    return merged_df


def load_basic_model_data():
    """Load basic model training history"""
    df = pd.read_csv("logs/nba_game_prediction/default/metrics.csv")
    # Merge train and val rows
    df = merge_train_val_rows(df)
    return df


def load_transformer_data():
    """Load Transformer training history"""
    df = pd.read_csv("logs/nba_transformer/transformer_default/metrics.csv")
    # Merge train and val rows
    df = merge_train_val_rows(df)
    return df


def load_transformer_with_teaminfo_data():
    """Load Transformer with TeamInfo training history"""
    df = pd.read_csv("logs/nba_transformer_with_teaminfo/transformer_with_teaminfo_default/metrics.csv")
    # Merge train and val rows
    df = merge_train_val_rows(df)
    return df


def plot_comparison():
    """Plot comparison of four models"""

    # Load data
    lr_data = load_logistic_regression_data()
    basic_data = load_basic_model_data()
    transformer_data = load_transformer_data()
    transformer_ti_data = load_transformer_with_teaminfo_data()

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Training Comparison of Four NBA Game Prediction Models", fontsize=18, fontweight="bold")

    # 1. Training Accuracy Comparison
    ax1 = axes[0, 0]
    ax1.plot(
        lr_data["iterations"],
        lr_data["train_acc"],
        label="Logistic Regression",
        linewidth=2,
        marker="o",
        markersize=3,
        markevery=5,
    )
    ax1.plot(
        basic_data["epoch"],
        basic_data["train_acc"],
        label="Basic Model",
        linewidth=2,
        marker="s",
        markersize=3,
        markevery=5,
    )
    ax1.plot(
        transformer_data["epoch"],
        transformer_data["train_acc"],
        label="Transformer",
        linewidth=2,
        marker="^",
        markersize=3,
        markevery=5,
    )
    ax1.plot(
        transformer_ti_data["epoch"],
        transformer_ti_data["train_acc"],
        label="Transformer+TeamInfo",
        linewidth=2,
        marker="d",
        markersize=3,
        markevery=5,
    )
    ax1.set_xlabel("Epoch/Iteration", fontsize=12)
    ax1.set_ylabel("Training Accuracy", fontsize=12)
    ax1.set_title("Training Accuracy Comparison", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.55, 0.72])

    # 2. Validation Accuracy Comparison
    ax2 = axes[0, 1]
    ax2.plot(
        lr_data["iterations"],
        lr_data["val_acc"],
        label="Logistic Regression",
        linewidth=2,
        marker="o",
        markersize=3,
        markevery=5,
    )
    ax2.plot(
        basic_data["epoch"],
        basic_data["val_acc"],
        label="Basic Model",
        linewidth=2,
        marker="s",
        markersize=3,
        markevery=5,
    )
    ax2.plot(
        transformer_data["epoch"],
        transformer_data["val_acc"],
        label="Transformer",
        linewidth=2,
        marker="^",
        markersize=3,
        markevery=5,
    )
    ax2.plot(
        transformer_ti_data["epoch"],
        transformer_ti_data["val_acc"],
        label="Transformer+TeamInfo",
        linewidth=2,
        marker="d",
        markersize=3,
        markevery=5,
    )
    ax2.set_xlabel("Epoch/Iteration", fontsize=12)
    ax2.set_ylabel("Validation Accuracy", fontsize=12)
    ax2.set_title("Validation Accuracy Comparison", fontsize=14, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.55, 0.75])

    # 3. Training Loss Comparison
    ax3 = axes[1, 0]
    ax3.plot(
        lr_data["iterations"],
        lr_data["train_loss"],
        label="Logistic Regression",
        linewidth=2,
        marker="o",
        markersize=3,
        markevery=5,
    )
    ax3.plot(
        basic_data["epoch"],
        basic_data["train_loss"],
        label="Basic Model",
        linewidth=2,
        marker="s",
        markersize=3,
        markevery=5,
    )
    ax3.plot(
        transformer_data["epoch"],
        transformer_data["train_loss"],
        label="Transformer",
        linewidth=2,
        marker="^",
        markersize=3,
        markevery=5,
    )
    ax3.plot(
        transformer_ti_data["epoch"],
        transformer_ti_data["train_loss"],
        label="Transformer+TeamInfo",
        linewidth=2,
        marker="d",
        markersize=3,
        markevery=5,
    )
    ax3.set_xlabel("Epoch/Iteration", fontsize=12)
    ax3.set_ylabel("Training Loss", fontsize=12)
    ax3.set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.60, 0.70])

    # 4. Validation Loss Comparison
    ax4 = axes[1, 1]
    ax4.plot(
        lr_data["iterations"],
        lr_data["val_loss"],
        label="Logistic Regression",
        linewidth=2,
        marker="o",
        markersize=3,
        markevery=5,
    )
    ax4.plot(
        basic_data["epoch"],
        basic_data["val_loss"],
        label="Basic Model",
        linewidth=2,
        marker="s",
        markersize=3,
        markevery=5,
    )
    ax4.plot(
        transformer_data["epoch"],
        transformer_data["val_loss"],
        label="Transformer",
        linewidth=2,
        marker="^",
        markersize=3,
        markevery=5,
    )
    ax4.plot(
        transformer_ti_data["epoch"],
        transformer_ti_data["val_loss"],
        label="Transformer+TeamInfo",
        linewidth=2,
        marker="d",
        markersize=3,
        markevery=5,
    )
    ax4.set_xlabel("Epoch/Iteration", fontsize=12)
    ax4.set_ylabel("Validation Loss", fontsize=12)
    ax4.set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.60, 0.70])

    plt.tight_layout()
    output_file = "model_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_file}")
    plt.close()

    # Print final performance comparison
    print("\n" + "=" * 80)
    print("Final Performance Comparison (at Epoch/Iteration 100/77)")
    print("=" * 80)

    print(f"\n{'Model':<30} {'Train Acc':<15} {'Val Acc':<15} {'Val Loss':<15}")
    print("-" * 80)

    lr_final = lr_data.iloc[-1]
    print(
        f"{'Logistic Regression':<30} {lr_final['train_acc']:.4f}{'':<10} {lr_final['val_acc']:.4f}{'':<10} {lr_final['val_loss']:.4f}"
    )

    basic_final = basic_data.iloc[-1]
    print(
        f"{'Basic Model':<30} {basic_final['train_acc']:.4f}{'':<10} {basic_final['val_acc']:.4f}{'':<10} {basic_final['val_loss']:.4f}"
    )

    transformer_final = transformer_data.iloc[-1]
    print(
        f"{'Transformer':<30} {transformer_final['train_acc']:.4f}{'':<10} {transformer_final['val_acc']:.4f}{'':<10} {transformer_final['val_loss']:.4f}"
    )

    transformer_ti_final = transformer_ti_data.iloc[-1]
    print(
        f"{'Transformer+TeamInfo':<30} {transformer_ti_final['train_acc']:.4f}{'':<10} {transformer_ti_final['val_acc']:.4f}{'':<10} {transformer_ti_final['val_loss']:.4f}"
    )

    print("-" * 80)

    # Calculate best validation accuracy
    print("\n" + "=" * 80)
    print("Best Validation Accuracy")
    print("=" * 80)

    lr_best_acc = lr_data["val_acc"].max()
    lr_best_epoch = lr_data["val_acc"].idxmax() + 1
    print(f"Logistic Regression: {lr_best_acc:.4f} (Iteration {lr_best_epoch})")

    basic_best_acc = basic_data["val_acc"].max()
    basic_best_epoch = basic_data[basic_data["val_acc"] == basic_best_acc]["epoch"].values[0]
    print(f"Basic Model: {basic_best_acc:.4f} (Epoch {int(basic_best_epoch)})")

    transformer_best_acc = transformer_data["val_acc"].max()
    transformer_best_epoch = transformer_data[transformer_data["val_acc"] == transformer_best_acc]["epoch"].values[0]
    print(f"Transformer: {transformer_best_acc:.4f} (Epoch {int(transformer_best_epoch)})")

    transformer_ti_best_acc = transformer_ti_data["val_acc"].max()
    transformer_ti_best_epoch = transformer_ti_data[transformer_ti_data["val_acc"] == transformer_ti_best_acc][
        "epoch"
    ].values[0]
    print(f"Transformer+TeamInfo: {transformer_ti_best_acc:.4f} (Epoch {int(transformer_ti_best_epoch)})")

    print("=" * 80)


if __name__ == "__main__":
    plot_comparison()

    print(f"\n{'='*80}")
    print("Visualization saved!")
    print(f"{'='*80}")
