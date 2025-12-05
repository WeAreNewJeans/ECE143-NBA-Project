import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from models.model_transformer import NBAGamePredictionTransformerModel
from dataloaders.dataloader import NBAGameDataModule
from extract_player_names import load_player_names


def calculate_feature_means(precomputed_dir):
    """
    Calculate mean feature values from all data (train+val+test).

    Returns:
        feature_means: numpy array of shape (240,) with mean values for each feature
    """
    train_team1 = np.load(os.path.join(precomputed_dir, 'train_team1_features.npy'))
    train_team2 = np.load(os.path.join(precomputed_dir, 'train_team2_features.npy'))

    val_team1 = np.load(os.path.join(precomputed_dir, 'val_team1_features.npy'))
    val_team2 = np.load(os.path.join(precomputed_dir, 'val_team2_features.npy'))

    test_team1 = np.load(os.path.join(precomputed_dir, 'test_team1_features.npy'))
    test_team2 = np.load(os.path.join(precomputed_dir, 'test_team2_features.npy'))

    all_features = np.concatenate([
        train_team1, train_team2,
        val_team1, val_team2,
        test_team1, test_team2
    ], axis=0)

    feature_means = all_features.mean(axis=0)

    return feature_means


def analyze_player_ablation_with_mean(model, team1_features, team2_features,
                                       feature_means, device='cpu', keep_sign=True):
    """
    Feature ablation analysis: Replace each player with mean feature values.

    Args:
        model: Trained model
        team1_features: shape (batch_size, 240)
        team2_features: shape (batch_size, 240)
        feature_means: shape (240,) - mean values for replacement
        device: Computing device
        keep_sign: If True, keep sign of impact (positive/negative)

    Returns:
        importance_scores: (batch_size, 12) - SIGNED importance for each player
            Positive: Removing player DECREASES win probability (good player)
            Negative: Removing player INCREASES win probability (bad player)
        abs_importance_scores: (batch_size, 12) - Absolute magnitude
    """
    model.eval()
    model.to(device)
    team1_features = team1_features.to(device)
    team2_features = team2_features.to(device)
    feature_means = torch.from_numpy(feature_means).float().to(device)

    batch_size = team1_features.size(0)
    num_players = 12
    player_dim = 20

    with torch.no_grad():
        original_logits = model(team1_features, team2_features)
        original_probs = torch.sigmoid(original_logits).squeeze()

        signed_importance_scores = torch.zeros(batch_size, num_players)
        abs_importance_scores = torch.zeros(batch_size, num_players)

        for player_idx in range(num_players):
            ablated_team1 = team1_features.clone()

            # Replace this player's features with mean
            start_idx = player_idx * player_dim
            end_idx = start_idx + player_dim
            ablated_team1[:, start_idx:end_idx] = feature_means[start_idx:end_idx]

            ablated_logits = model(ablated_team1, team2_features)
            ablated_probs = torch.sigmoid(ablated_logits).squeeze()

            # Calculate prediction change
            # Positive = removing player DECREASES win prob (player helps team)
            # Negative = removing player INCREASES win prob (player hurts team)
            prob_change = original_probs - ablated_probs

            signed_importance_scores[:, player_idx] = prob_change.cpu()
            abs_importance_scores[:, player_idx] = torch.abs(prob_change).cpu()

    if keep_sign:
        return signed_importance_scores, abs_importance_scores
    else:
        return abs_importance_scores, abs_importance_scores


def extract_season_from_date(date_str):
    """Extract season from game date."""
    import pandas as pd
    date = pd.to_datetime(date_str)
    year = date.year
    month = date.month

    if month <= 6:
        return f"{year-1}-{year}"
    else:
        return f"{year}-{year+1}"


def analyze_all_splits(model, precomputed_dir, player_mappings, feature_means, device='cpu'):
    """
    Analyze all data splits (train + val + test) for maximum coverage.

    Returns:
        player_season_importance: Dict mapping (player_name, season) -> list of importance scores
    """
    model.eval()
    model.to(device)

    player_season_importance = defaultdict(lambda: {'signed_scores': [], 'abs_scores': [], 'count': 0})

    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"Processing {split} split")

        team1_features = np.load(os.path.join(precomputed_dir, f'{split}_team1_features.npy'))
        team2_features = np.load(os.path.join(precomputed_dir, f'{split}_team2_features.npy'))
        print(f"Loaded {len(team1_features)} games from {split} set")

        team1_tensor = torch.from_numpy(team1_features).float()
        team2_tensor = torch.from_numpy(team2_features).float()

        batch_size = 32
        num_batches = (len(team1_tensor) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc=f"Analyzing {split}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(team1_tensor))

            batch_team1 = team1_tensor[start_idx:end_idx]
            batch_team2 = team2_tensor[start_idx:end_idx]

            # Analyze Team1 (home) importance
            signed_importance_team1, abs_importance_team1 = analyze_player_ablation_with_mean(
                model, batch_team1, batch_team2, feature_means, device, keep_sign=True
            )

            # Analyze Team2 (away) importance - swap team1 and team2 positions
            signed_importance_team2, abs_importance_team2 = analyze_player_ablation_with_mean(
                model, batch_team2, batch_team1, feature_means, device, keep_sign=True
            )

            for i in range(len(batch_team1)):
                game_idx = start_idx + i

                if game_idx >= len(player_mappings[split]):
                    continue

                game_data = player_mappings[split][game_idx]

                season = extract_season_from_date(game_data['game_date'])

                # Analyze both Team1 (home) and Team2 (away) players
                team1_players = game_data['team1_players']
                team2_players = game_data['team2_players']

                # Team1 players (home team)
                for player_pos in range(12):
                    player_id, player_name = team1_players[player_pos]

                    if player_name != 'UNKNOWN' and player_name is not None:
                        player_season_key = (player_name, season)
                        signed_score = signed_importance_team1[i, player_pos].item()
                        abs_score = abs_importance_team1[i, player_pos].item()

                        player_season_importance[player_season_key]['signed_scores'].append(signed_score)
                        player_season_importance[player_season_key]['abs_scores'].append(abs_score)
                        player_season_importance[player_season_key]['count'] += 1

                # Team2 players (away team)
                for player_pos in range(12):
                    player_id, player_name = team2_players[player_pos]

                    if player_name != 'UNKNOWN' and player_name is not None:
                        player_season_key = (player_name, season)
                        signed_score = signed_importance_team2[i, player_pos].item()
                        abs_score = abs_importance_team2[i, player_pos].item()

                        player_season_importance[player_season_key]['signed_scores'].append(signed_score)
                        player_season_importance[player_season_key]['abs_scores'].append(abs_score)
                        player_season_importance[player_season_key]['count'] += 1

    return player_season_importance


def create_results_dataframe(player_season_importance, min_games=30):
    """
    Convert player-season importance to DataFrame.

    Args:
        player_season_importance: Dict from analyze_all_splits
        min_games: Minimum games required to be included

    Returns:
        results_df: DataFrame with player-season analysis
    """
    results = []

    for (player_name, season), data in player_season_importance.items():
        games_count = data['count']

        if games_count < min_games:
            continue

        signed_scores = data['signed_scores']

        mean_signed = np.mean(signed_scores)
        std_signed = np.std(signed_scores)
        median_signed = np.median(signed_scores)

        # Positive games: player helps team (removing them decreases win prob)
        positive_games = sum(1 for s in signed_scores if s > 0)
        negative_games = sum(1 for s in signed_scores if s < 0)
        positive_rate = positive_games / games_count if games_count > 0 else 0

        results.append({
            'Player_Name': player_name,
            'Season': season,
            'Player_Season': f"{player_name} ({season})",
            'Games_Analyzed': games_count,
            'Mean_Signed_Impact': mean_signed,
            'Std_Signed_Impact': std_signed,
            'Median_Signed_Impact': median_signed,
            'Positive_Games': positive_games,
            'Negative_Games': negative_games,
            'Positive_Rate': positive_rate,
        })

    df = pd.DataFrame(results)
    df = df.sort_values('Mean_Signed_Impact', ascending=False).reset_index(drop=True)
    return df


def visualize_top_player_seasons(results_df, top_n=30, save_path='top_player_seasons.png'):
    """
    Visualize top player-season combinations with signed impact.

    Args:
        results_df: Results DataFrame
        top_n: Number of top player-seasons to show
        save_path: Path to save visualization
    """
    top_players = results_df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 10))

    values = top_players['Mean_Signed_Impact'].values[::-1]
    colors = ['green' if v > 0 else 'red' for v in values]

    y_pos = np.arange(top_n)
    ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_players['Player_Season'].values[::-1], fontsize=9)
    ax.set_xlabel('Signed Impact (Positive = Helpful, Negative = Harmful)', fontsize=11)
    ax.set_title(f'Top {top_n} Player',
                  fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    # Add positive rate annotations (percentage = how often player helps team)
    for i, (idx, row) in enumerate(top_players.iterrows()):
        pos_rate = row['Positive_Rate'] * 100
        ax.text(row['Mean_Signed_Impact'] + 0.001 if row['Mean_Signed_Impact'] > 0 else row['Mean_Signed_Impact'] - 0.001,
                 top_n - i - 1,
                 f"{pos_rate:.0f}%",
                 va='center', ha='left' if row['Mean_Signed_Impact'] > 0 else 'right',
                 fontsize=8, alpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def visualize_player_across_seasons(results_df, player_name, save_path=None):
    """
    Visualize a single player's importance across different seasons.

    Args:
        results_df: Results DataFrame
        player_name: Name of player to analyze
        save_path: Path to save (if None, auto-generate)
    """
    player_data = results_df[results_df['Player_Name'] == player_name].copy()

    if len(player_data) == 0:
        print(f"No data found for {player_name}")
        return

    player_data = player_data.sort_values('Season')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Signed Impact over seasons
    ax1.plot(player_data['Season'], player_data['Mean_Signed_Impact'],
             marker='o', linewidth=2, markersize=8, color='steelblue', label='Signed Impact')
    ax1.fill_between(range(len(player_data)),
                     player_data['Mean_Signed_Impact'] - player_data['Std_Signed_Impact'],
                     player_data['Mean_Signed_Impact'] + player_data['Std_Signed_Impact'],
                     alpha=0.3, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Neutral')
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Signed Impact (Positive = Helpful)', fontsize=12)
    ax1.set_title(f'{player_name} - Impact Over Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Games analyzed
    ax2.bar(player_data['Season'], player_data['Games_Analyzed'],
            color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Games Analyzed', fontsize=12)
    ax2.set_title(f'{player_name} - Data Coverage', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path is None:
        safe_name = player_name.replace(' ', '_').replace('.', '')
        save_path = f'player_seasons_{safe_name}.png'

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved player analysis to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Player Importance by Season (Ablation Method)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/last.ckpt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--precomputed_dir',
        type=str,
        default='data/precomputed_features',
        help='Directory with precomputed features'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='analysis_results',
        help='Output directory'
    )
    parser.add_argument(
        '--min_games',
        type=int,
        default=30,
        help='Minimum games required for a player-season to be included'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=50,
        help='Number of top player-seasons to visualize'
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Player-Season Importance Analysis (Ablation Method)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Analyzing all splits (train + val + test)")
    print(f"Min games required for a player-season to be included: {args.min_games}")

    player_names_path = os.path.join(args.precomputed_dir, 'player_names.pkl')
    if not os.path.exists(player_names_path):
        print("ERROR: Player names file not found!")
        print(f"\nFile not found: {player_names_path}")
        print("\nPlease run: python extract_player_names.py")
        exit(1)

    player_mappings = load_player_names(args.precomputed_dir)

    model = NBAGamePredictionTransformerModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(args.device)

    feature_means = calculate_feature_means(args.precomputed_dir)

    player_season_importance = analyze_all_splits(
        model, args.precomputed_dir, player_mappings,
        feature_means, device=args.device
    )

    results_df = create_results_dataframe(player_season_importance, min_games=args.min_games)

    print(f"\nTotal unique player-season combinations: {len(player_season_importance)}")
    print(f"After filtering (min {args.min_games} games): {len(results_df)}")

    results_path = os.path.join(args.output_dir, 'player_season_importance.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 80)
    print(f"Top {args.top_n} Combinations")
    print(results_df.head(args.top_n).to_string(index=False))

    vis_path = os.path.join(args.output_dir, f'top_{args.top_n}_player_seasons.png')
    visualize_top_player_seasons(results_df, top_n=args.top_n, save_path=vis_path)

    print(f"\nDnoe! All results saved to: {args.output_dir}/")

if __name__ == '__main__':
    main()
