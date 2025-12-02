"""
Advanced Model Analysis and Comparison
深入分析和比较四个NBA比赛预测模型
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = "model_analysis_results"


def create_output_directory():
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(OUTPUT_DIR) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    # Also create a 'latest' symlink for convenience
    latest_path = Path(OUTPUT_DIR) / "latest"
    if latest_path.exists():
        if latest_path.is_symlink():
            latest_path.unlink()
        else:
            import shutil

            shutil.rmtree(latest_path)

    try:
        latest_path.symlink_to(timestamp, target_is_directory=True)
    except:
        # On Windows, symlinks may not work, so just copy path
        pass

    print(f"\nOutput directory: {output_path}")
    return output_path


def merge_train_val_rows(df):
    """Merge training and validation rows by epoch"""
    train_df = df[df["train_acc"].notna()][["epoch", "train_loss", "train_acc"]].copy()
    val_df = df[df["val_acc"].notna()][["epoch", "val_loss", "val_acc"]].copy()
    merged_df = pd.merge(train_df, val_df, on="epoch", how="outer")
    merged_df = merged_df.sort_values("epoch").reset_index(drop=True)
    return merged_df


def load_all_data():
    """Load all model data"""
    data = {}

    # Logistic Regression
    lr_df = pd.read_csv("LR_results/baseline_logistic_training_history.csv")
    lr_df = lr_df[:77]  # Remove duplicates
    lr_df["epoch"] = lr_df["iterations"]
    data["Logistic Regression"] = lr_df

    # Basic Model
    basic_df = pd.read_csv("logs/nba_game_prediction/default/metrics.csv")
    data["Basic Model"] = merge_train_val_rows(basic_df)

    # Transformer
    trans_df = pd.read_csv("logs/nba_transformer/transformer_default/metrics.csv")
    data["Transformer"] = merge_train_val_rows(trans_df)

    # Transformer + TeamInfo
    trans_ti_df = pd.read_csv("logs/nba_transformer_with_teaminfo/transformer_with_teaminfo_default/metrics.csv")
    data["Transformer+TeamInfo"] = merge_train_val_rows(trans_ti_df)

    return data


def analyze_overfitting(data, output_dir):
    """
    1. Overfitting Analysis
    Analyze train-val gap to detect overfitting
    """
    print("\n" + "=" * 80)
    print("1. OVERFITTING ANALYSIS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Overfitting Analysis: Train-Val Gap", fontsize=16, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, 4))

    for idx, (name, df) in enumerate(data.items()):
        ax = axes[idx // 2, idx % 2]

        # Calculate gaps
        acc_gap = df["train_acc"] - df["val_acc"]
        loss_gap = df["val_loss"] - df["train_loss"]

        # Plot
        ax2 = ax.twinx()
        line1 = ax.plot(
            df["epoch"], acc_gap, "o-", label="Accuracy Gap", color=colors[0], linewidth=2, markersize=4, markevery=5
        )
        line2 = ax2.plot(
            df["epoch"], loss_gap, "s-", label="Loss Gap", color=colors[1], linewidth=2, markersize=4, markevery=5
        )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Train Acc - Val Acc", fontsize=11, color=colors[0])
        ax2.set_ylabel("Val Loss - Train Loss", fontsize=11, color=colors[1])
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="y", labelcolor=colors[0])
        ax2.tick_params(axis="y", labelcolor=colors[1])

        # Add reference line at 0
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Calculate final gap
        final_acc_gap = acc_gap.iloc[-1]
        final_loss_gap = loss_gap.iloc[-1]

        # Add text box with statistics
        textstr = f"Final Gaps:\nAcc: {final_acc_gap:.4f}\nLoss: {final_loss_gap:.4f}"
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Print statistics
        print(f"\n{name}:")
        print(f"  Final Accuracy Gap (Train-Val): {final_acc_gap:.4f}")
        print(f"  Final Loss Gap (Val-Train): {final_loss_gap:.4f}")
        print(f"  Max Accuracy Gap: {acc_gap.max():.4f}")
        print(f"  Overfitting Trend: {'Increasing' if acc_gap.iloc[-1] > acc_gap.iloc[10] else 'Stable/Decreasing'}")

    plt.tight_layout()
    output_file = output_dir / "analysis_1_overfitting.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_file}")
    plt.close()


def analyze_convergence(data, output_dir):
    """
    2. Convergence Analysis
    Analyze convergence speed and stability
    """
    print("\n" + "=" * 80)
    print("2. CONVERGENCE ANALYSIS")
    print("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Convergence Analysis", fontsize=16, fontweight="bold")

    # Left: Validation accuracy improvement rate
    ax1 = axes[0]
    for name, df in data.items():
        val_acc = df["val_acc"].values
        # Calculate moving average to smooth
        window = 5
        smoothed = pd.Series(val_acc).rolling(window=window, min_periods=1).mean()
        ax1.plot(df["epoch"], smoothed, label=name, linewidth=2.5, alpha=0.8)

    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Validation Accuracy (Smoothed)", fontsize=12, fontweight="bold")
    ax1.set_title("Convergence Speed", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Learning rate (epochs to reach certain accuracy)
    ax2 = axes[1]
    milestones = [0.60, 0.62, 0.64, 0.66]
    x_pos = np.arange(len(data))
    width = 0.2

    for i, milestone in enumerate(milestones):
        epochs_to_reach = []
        for name, df in data.items():
            epochs = df[df["val_acc"] >= milestone]["epoch"].values
            if len(epochs) > 0:
                epochs_to_reach.append(epochs[0])
            else:
                epochs_to_reach.append(100)  # Didn't reach

        ax2.bar(x_pos + i * width, epochs_to_reach, width, label=f"Acc ≥ {milestone:.2f}", alpha=0.8)

    ax2.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Epochs to Reach", fontsize=12, fontweight="bold")
    ax2.set_title("Speed to Reach Accuracy Milestones", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos + width * 1.5)
    ax2.set_xticklabels(data.keys(), rotation=15, ha="right")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / "analysis_2_convergence.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_file}")
    plt.close()

    # Print statistics
    for name, df in data.items():
        print(f"\n{name}:")
        val_acc = df["val_acc"].values
        # Epochs to reach 60% accuracy
        epoch_60 = df[df["val_acc"] >= 0.60]["epoch"].values
        if len(epoch_60) > 0:
            print(f"  Epochs to reach 60% acc: {epoch_60[0]:.0f}")
        else:
            print(f"  Epochs to reach 60% acc: >100")

        # Improvement in first 10 epochs
        if len(val_acc) >= 10:
            improvement = val_acc[9] - val_acc[0]
            print(f"  Improvement in first 10 epochs: {improvement:.4f}")


def analyze_stability(data, output_dir):
    """
    3. Training Stability Analysis
    Analyze variance and oscillation in training
    """
    print("\n" + "=" * 80)
    print("3. TRAINING STABILITY ANALYSIS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Training Stability Analysis", fontsize=16, fontweight="bold")

    for idx, (name, df) in enumerate(data.items()):
        ax = axes[idx // 2, idx % 2]

        # Calculate rolling standard deviation (volatility)
        window = 10
        val_acc_std = df["val_acc"].rolling(window=window, min_periods=1).std()
        val_loss_std = df["val_loss"].rolling(window=window, min_periods=1).std()

        ax2 = ax.twinx()
        line1 = ax.plot(
            df["epoch"], val_acc_std, "o-", label="Val Acc Std", linewidth=2, markersize=3, markevery=5, alpha=0.8
        )
        line2 = ax2.plot(
            df["epoch"], val_loss_std, "s-", label="Val Loss Std", linewidth=2, markersize=3, markevery=5, alpha=0.8
        )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Val Acc Std (10-epoch window)", fontsize=10)
        ax2.set_ylabel("Val Loss Std (10-epoch window)", fontsize=10)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=9)

        # Print statistics
        print(f"\n{name}:")
        print(f"  Avg Val Acc Std (last 20 epochs): {val_acc_std.iloc[-20:].mean():.6f}")
        print(f"  Avg Val Loss Std (last 20 epochs): {val_loss_std.iloc[-20:].mean():.6f}")
        print(f"  Stability Score (lower is better): {val_acc_std.iloc[-20:].mean():.6f}")

    plt.tight_layout()
    output_file = output_dir / "analysis_3_stability.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_file}")
    plt.close()


def analyze_performance_summary(data, output_dir):
    """
    4. Performance Summary Table
    Create comprehensive performance comparison
    """
    print("\n" + "=" * 80)
    print("4. COMPREHENSIVE PERFORMANCE SUMMARY")
    print("=" * 80)

    summary = []

    for name, df in data.items():
        metrics = {
            "Model": name,
            "Best Val Acc": df["val_acc"].max(),
            "Best Val Acc Epoch": df.loc[df["val_acc"].idxmax(), "epoch"],
            "Final Val Acc": df["val_acc"].iloc[-1],
            "Best Val Loss": df["val_loss"].min(),
            "Final Val Loss": df["val_loss"].iloc[-1],
            "Final Train Acc": df["train_acc"].iloc[-1],
            "Overfit Gap": df["train_acc"].iloc[-1] - df["val_acc"].iloc[-1],
            "Acc Improvement": df["val_acc"].iloc[-1] - df["val_acc"].iloc[0],
        }
        summary.append(metrics)

    summary_df = pd.DataFrame(summary)

    # Print table
    print("\n" + summary_df.to_string(index=False))

    # Save to CSV
    csv_file = output_dir / "analysis_4_performance_summary.csv"
    summary_df.to_csv(csv_file, index=False)
    print(f"\nSaved: {csv_file}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Performance Metrics Comparison", fontsize=16, fontweight="bold")

    models = summary_df["Model"].values
    x_pos = np.arange(len(models))

    # 1. Best vs Final Validation Accuracy
    ax1 = axes[0, 0]
    width = 0.35
    ax1.bar(x_pos - width / 2, summary_df["Best Val Acc"], width, label="Best Val Acc", alpha=0.8)
    ax1.bar(x_pos + width / 2, summary_df["Final Val Acc"], width, label="Final Val Acc", alpha=0.8)
    ax1.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax1.set_title("Best vs Final Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Overfitting Gap
    ax2 = axes[0, 1]
    colors = ["green" if x < 0.05 else "orange" if x < 0.10 else "red" for x in summary_df["Overfit Gap"]]
    ax2.bar(x_pos, summary_df["Overfit Gap"], color=colors, alpha=0.7)
    ax2.set_ylabel("Train Acc - Val Acc", fontsize=11, fontweight="bold")
    ax2.set_title("Overfitting Gap (Lower is Better)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=15, ha="right")
    ax2.axhline(y=0.05, color="orange", linestyle="--", alpha=0.5, label="Moderate (0.05)")
    ax2.axhline(y=0.10, color="red", linestyle="--", alpha=0.5, label="High (0.10)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Best Epoch
    ax3 = axes[1, 0]
    ax3.bar(x_pos, summary_df["Best Val Acc Epoch"], alpha=0.8, color="skyblue")
    ax3.set_ylabel("Epoch", fontsize=11, fontweight="bold")
    ax3.set_title("Epoch Achieving Best Val Acc", fontsize=12, fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=15, ha="right")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Accuracy Improvement
    ax4 = axes[1, 1]
    ax4.bar(x_pos, summary_df["Acc Improvement"], alpha=0.8, color="lightcoral")
    ax4.set_ylabel("Accuracy Improvement", fontsize=11, fontweight="bold")
    ax4.set_title("Total Validation Accuracy Improvement", fontsize=12, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models, rotation=15, ha="right")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / "analysis_4_performance_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close()


def analyze_learning_dynamics(data, output_dir):
    """
    5. Learning Dynamics
    Analyze the rate of improvement over time
    """
    print("\n" + "=" * 80)
    print("5. LEARNING DYNAMICS ANALYSIS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Learning Dynamics: Rate of Improvement", fontsize=16, fontweight="bold")

    for idx, (name, df) in enumerate(data.items()):
        ax = axes[idx // 2, idx % 2]

        # Calculate gradient (rate of change) of validation accuracy
        val_acc_diff = df["val_acc"].diff()

        # Plot
        ax.plot(df["epoch"].iloc[1:], val_acc_diff.iloc[1:], "o-", linewidth=2, markersize=3, alpha=0.7)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Val Acc Change (Δ)", fontsize=11)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add statistics
        avg_improvement_early = val_acc_diff.iloc[1:11].mean()
        avg_improvement_late = val_acc_diff.iloc[-10:].mean()

        textstr = f"Early (1-10): {avg_improvement_early:.5f}\nLate (last 10): {avg_improvement_late:.5f}"
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        )

        print(f"\n{name}:")
        print(f"  Avg improvement rate (first 10 epochs): {avg_improvement_early:.5f}")
        print(f"  Avg improvement rate (last 10 epochs): {avg_improvement_late:.5f}")
        print(f"  Total epochs with improvement: {(val_acc_diff > 0).sum()}")
        print(f"  Total epochs with decline: {(val_acc_diff < 0).sum()}")

    plt.tight_layout()
    output_file = output_dir / "analysis_5_learning_dynamics.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_file}")
    plt.close()


def analyze_final_ranking(data, output_dir):
    """
    6. Final Ranking
    Rank models by different criteria
    """
    print("\n" + "=" * 80)
    print("6. MODEL RANKING BY DIFFERENT CRITERIA")
    print("=" * 80)

    criteria = []

    for name, df in data.items():
        # Calculate various metrics
        best_val_acc = df["val_acc"].max()
        final_val_acc = df["val_acc"].iloc[-1]
        best_val_loss = df["val_loss"].min()
        final_val_loss = df["val_loss"].iloc[-1]
        overfit_gap = df["train_acc"].iloc[-1] - df["val_acc"].iloc[-1]
        val_acc_std = df["val_acc"].rolling(window=10).std().iloc[-10:].mean()

        criteria.append(
            {
                "Model": name,
                "Best Val Acc": best_val_acc,
                "Final Val Acc": final_val_acc,
                "Best Val Loss": best_val_loss,
                "Final Val Loss": final_val_loss,
                "Overfit Gap": overfit_gap,
                "Stability": val_acc_std,
            }
        )

    df_rank = pd.DataFrame(criteria)

    print("\n### Ranking by Best Validation Accuracy ###")
    ranked_best = df_rank.sort_values("Best Val Acc", ascending=False)
    for i, (idx, row) in enumerate(ranked_best.iterrows(), 1):
        print(f"{i}. {row['Model']}: {row['Best Val Acc']:.4f}")

    print("\n### Ranking by Final Validation Accuracy ###")
    ranked_final = df_rank.sort_values("Final Val Acc", ascending=False)
    for i, (idx, row) in enumerate(ranked_final.iterrows(), 1):
        print(f"{i}. {row['Model']}: {row['Final Val Acc']:.4f}")

    print("\n### Ranking by Generalization (Lower Overfit Gap) ###")
    ranked_gen = df_rank.sort_values("Overfit Gap", ascending=True)
    for i, (idx, row) in enumerate(ranked_gen.iterrows(), 1):
        print(f"{i}. {row['Model']}: {row['Overfit Gap']:.4f}")

    print("\n### Ranking by Stability (Lower Variance) ###")
    ranked_stab = df_rank.sort_values("Stability", ascending=True)
    for i, (idx, row) in enumerate(ranked_stab.iterrows(), 1):
        print(f"{i}. {row['Model']}: {row['Stability']:.6f}")

    # Create ranking visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create ranking matrix
    ranking_matrix = []
    categories = ["Best Val Acc", "Final Val Acc", "Generalization", "Stability"]

    for model in df_rank["Model"]:
        ranks = []
        # Best Val Acc
        ranks.append(df_rank.sort_values("Best Val Acc", ascending=False)["Model"].tolist().index(model) + 1)
        # Final Val Acc
        ranks.append(df_rank.sort_values("Final Val Acc", ascending=False)["Model"].tolist().index(model) + 1)
        # Generalization
        ranks.append(df_rank.sort_values("Overfit Gap", ascending=True)["Model"].tolist().index(model) + 1)
        # Stability
        ranks.append(df_rank.sort_values("Stability", ascending=True)["Model"].tolist().index(model) + 1)
        ranking_matrix.append(ranks)

    ranking_matrix = np.array(ranking_matrix)

    # Plot heatmap
    im = ax.imshow(ranking_matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=4)

    # Set ticks
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(df_rank)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(df_rank["Model"])

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(df_rank)):
        for j in range(len(categories)):
            text = ax.text(
                j, i, int(ranking_matrix[i, j]), ha="center", va="center", color="black", fontsize=12, fontweight="bold"
            )

    ax.set_title("Model Rankings Heatmap (1=Best, 4=Worst)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Rank")

    plt.tight_layout()
    output_file = output_dir / "analysis_6_ranking.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_file}")
    plt.close()


def main():
    """Run all analyses"""
    print("=" * 80)
    print("ADVANCED MODEL ANALYSIS")
    print("Comprehensive comparison of 4 NBA game prediction models")
    print("=" * 80)

    # Create output directory
    output_dir = create_output_directory()

    # Load data
    print("\nLoading data...")
    data = load_all_data()
    print(f"Loaded {len(data)} models")

    # Run analyses
    analyze_overfitting(data, output_dir)
    analyze_convergence(data, output_dir)
    analyze_stability(data, output_dir)
    analyze_performance_summary(data, output_dir)
    analyze_learning_dynamics(data, output_dir)
    analyze_final_ranking(data, output_dir)

    # Create README file
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# NBA Model Analysis Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Files\n\n")
        f.write("1. `analysis_1_overfitting.png` - Overfitting analysis (Train-Val gap)\n")
        f.write("2. `analysis_2_convergence.png` - Convergence speed comparison\n")
        f.write("3. `analysis_3_stability.png` - Training stability analysis\n")
        f.write("4. `analysis_4_performance_comparison.png` - Performance metrics comparison\n")
        f.write("5. `analysis_4_performance_summary.csv` - Summary statistics table\n")
        f.write("6. `analysis_5_learning_dynamics.png` - Learning rate dynamics\n")
        f.write("7. `analysis_6_ranking.png` - Model rankings heatmap\n")
        f.write("\n## Models Analyzed\n\n")
        f.write("- Logistic Regression (Baseline)\n")
        f.write("- Basic Model (Neural Network)\n")
        f.write("- Transformer\n")
        f.write("- Transformer + TeamInfo\n")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"Quick access via: {OUTPUT_DIR}/latest/")
    print("\nGenerated files:")
    print("  1. analysis_1_overfitting.png - Overfitting analysis")
    print("  2. analysis_2_convergence.png - Convergence speed")
    print("  3. analysis_3_stability.png - Training stability")
    print("  4. analysis_4_performance_comparison.png - Performance metrics")
    print("  5. analysis_4_performance_summary.csv - Summary table")
    print("  6. analysis_5_learning_dynamics.png - Learning rate over time")
    print("  7. analysis_6_ranking.png - Model rankings")
    print("  8. README.md - Analysis documentation")
    print("=" * 80)


if __name__ == "__main__":
    main()
