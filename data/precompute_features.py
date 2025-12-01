import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle


def extract_season(date_str):
    """Extract NBA season from date string."""
    date = pd.to_datetime(date_str)
    year = date.year
    month = date.month

    if month <= 6:
        return f"{year - 1}-{year}"
    else:
        return f"{year}-{year + 1}"


def precompute_all_features(
    team_stats_path,
    season_player_path,
    player_stats_path,
    output_dir,
    num_recent_games=5,
    train_ratio=0.7,
    val_ratio=0.15,
    random_seed=42
):
    """
    Pre-compute all game features and save to disk.

    Args:
        team_stats_path: Path to TeamStatistics.csv
        season_player_path: Path to Season_player_data.csv
        player_stats_path: Path to PlayerStatistics.csv
        output_dir: Directory to save pre-computed features
        num_recent_games: Number of recent games for rolling stats
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)

    print("="*80)
    print("Pre-computing NBA Game Features")
    print("="*80)
    print(f"Random seed: {random_seed}")

    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading data files...")
    team_stats_df = pd.read_csv(team_stats_path)
    season_stats_df = pd.read_csv(season_player_path)
    player_stats_df = pd.read_csv(player_stats_path, low_memory=False)

    print(f"Loaded {len(team_stats_df)} team game records")
    print(f"Loaded {len(season_stats_df)} player-season records")
    print(f"Loaded {len(player_stats_df)} player game records")

    print("\nProcessing team stats...")
    team_stats_df['season'] = team_stats_df['gameDateTimeEst'].apply(extract_season)

    # Filter data from 2000-2001 season onwards
    team_stats_df['season_start_year'] = team_stats_df['season'].apply(
        lambda x: int(x.split('-')[0])
    )
    team_stats_df = team_stats_df[team_stats_df['season_start_year'] >= 2000].copy()
    print(f"Filtered to games from 2000-2001 season onwards: {len(team_stats_df)} records")

    # Keep only one record per game (home team perspective)
    games_df = team_stats_df[team_stats_df['home'] == 1].copy()
    print(f"Total unique games: {len(games_df)}")

    player_stats_df['gameDateTimeEst'] = pd.to_datetime(
        player_stats_df['gameDateTimeEst'],
        format='ISO8601',
        utc=True
    ).dt.tz_localize(None)

    games_df['gameDateTimeEst'] = pd.to_datetime(
        games_df['gameDateTimeEst'],
        format='ISO8601',
        utc=True
    ).dt.tz_localize(None)

    # Sort by date
    games_df = games_df.sort_values('gameDateTimeEst').reset_index(drop=True)

    # Split data by season
    # Train+Val: 2000-2001 to 2023-2024 (ratio 9:1)
    # Test: 2024-2025
    # Exclude: 2025-2026 (ongoing)
    print("\nSplitting data by season...")

    games_df = games_df[games_df['season'] != '2025-2026'].copy()
    print(f"After excluding 2025-2026 season: {len(games_df)} games")

    train_val_df = games_df[games_df['season'] < '2024-2025'].copy()
    test_df = games_df[games_df['season'] == '2024-2025'].copy()

    # Randomly shuffle train_val and split into train and val (9:1 ratio)
    train_val_df = train_val_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    n_train_val = len(train_val_df)
    n_train = int(n_train_val * 0.9)

    train_df = train_val_df.iloc[:n_train]
    val_df = train_val_df.iloc[n_train:]

    print(f"\nData split by season:")
    print(f"  Training: {len(train_df)} games (2000-2001 to 2023-2024, random 90%)")
    print(f"  Validation: {len(val_df)} games (2000-2001 to 2023-2024, random 10%)")
    print(f"  Test: {len(test_df)} games (2024-2025 season)")

    print(f"\nSeason ranges:")
    print(f"  Training: {train_df['season'].min()} to {train_df['season'].max()}")
    print(f"  Validation: {val_df['season'].min()} to {val_df['season'].max()}")
    print(f"  Test: {test_df['season'].min()} to {test_df['season'].max()}")

    stat_columns = [
        'Season_PPG', 'Season_RPG', 'Season_APG', 'Season_SPG', 'Season_BPG',
        'Season_TOV', 'Season_MPG', 'Season_FG%', 'Season_3P%', 'Season_FT%'
    ]

    season_stats_index = season_stats_df.set_index(['personId', 'season'])
    player_stats_df = player_stats_df.sort_values('gameDateTimeEst')

    game_players_index = {}
    for game_id in tqdm(games_df['gameId'].unique(), desc="Indexing games"):
        game_data = player_stats_df[player_stats_df['gameId'] == game_id]
        game_players_index[game_id] = game_data[
            ['personId', 'playerteamName', 'numMinutes']
        ].to_dict('records')

    team_id_to_name = dict(zip(games_df['teamId'], games_df['teamName']))

    def get_season_stats(player_id, season):
        """Get season average stats for a player."""
        try:
            stats = season_stats_index.loc[(player_id, season)]
            if isinstance(stats, pd.DataFrame):
                stats = stats.iloc[0]
            values = np.array([stats[col] for col in stat_columns], dtype=np.float32)
        except KeyError:
            values = np.zeros(10, dtype=np.float32)

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values

    def get_recent_stats(player_id, game_date):
        """Get recent N-game average stats for a player."""
        recent_games = player_stats_df[
            (player_stats_df['personId'] == player_id) &
            (player_stats_df['gameDateTimeEst'] < game_date)
        ].tail(num_recent_games)

        if len(recent_games) == 0:
            return np.zeros(10, dtype=np.float32)

        stat_mapping = {
            'Season_PPG': 'points',
            'Season_RPG': 'reboundsTotal',
            'Season_APG': 'assists',
            'Season_SPG': 'steals',
            'Season_BPG': 'blocks',
            'Season_TOV': 'turnovers',
            'Season_MPG': 'numMinutes',
            'Season_FG%': 'fieldGoalsPercentage',
            'Season_3P%': 'threePointersPercentage',
            'Season_FT%': 'freeThrowsPercentage'
        }

        recent_stats = []
        for col in stat_columns:
            game_col = stat_mapping[col]
            recent_stats.append(recent_games[game_col].mean())

        values = np.array(recent_stats, dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return values

    def get_team_features(team_id, game_id, season, game_date):
        """
        Get features for a team (12 players * 20 features = 240-dim vector).

        Note: Players are sorted by playing time (numMinutes) in descending order.
        """
        team_name = team_id_to_name.get(team_id)

        game_players = [
            p for p in game_players_index.get(game_id, [])
            if p['playerteamName'] == team_name
        ]

        # Sort by playing time (descending) and take top 12
        game_players = sorted(game_players, key=lambda x: x['numMinutes'], reverse=True)[:12]

        player_features = []
        for player in game_players:
            person_id = player['personId']
            season_stats = get_season_stats(person_id, season)
            recent_stats = get_recent_stats(person_id, game_date)
            player_feature = np.concatenate([season_stats, recent_stats])
            player_features.append(player_feature)

        # Pad with zeros if less than 12 players
        while len(player_features) < 12:
            player_features.append(np.zeros(20))

        # (12, 20) -> (240,)
        team_features = np.vstack(player_features).flatten()
        team_features = np.nan_to_num(team_features, nan=0.0, posinf=0.0, neginf=0.0)

        return team_features

    def compute_split_features(df, split_name):
        """Compute features for a data split."""
        print(f"\n{'='*80}")
        print(f"Computing {split_name} features...")
        print(f"{'='*80}")

        team1_features_list = []
        team2_features_list = []
        labels_list = []

        for idx in tqdm(range(len(df)), desc=f"Computing {split_name}"):
            game = df.iloc[idx]

            team1_features = get_team_features(
                game['teamId'],
                game['gameId'],
                game['season'],
                game['gameDateTimeEst']
            )

            team2_features = get_team_features(
                game['opponentTeamId'],
                game['gameId'],
                game['season'],
                game['gameDateTimeEst']
            )

            label = float(game['win'])

            team1_features_list.append(team1_features)
            team2_features_list.append(team2_features)
            labels_list.append(label)

        team1_features = np.vstack(team1_features_list).astype(np.float32)
        team2_features = np.vstack(team2_features_list).astype(np.float32)
        labels = np.array(labels_list, dtype=np.float32)

        print(f"{split_name} features computed:")
        print(f"  Team1 shape: {team1_features.shape}")
        print(f"  Team2 shape: {team2_features.shape}")
        print(f"  Labels shape: {labels.shape}")

        return team1_features, team2_features, labels

    # training features
    train_team1, train_team2, train_labels = compute_split_features(train_df, "Training")
    np.save(os.path.join(output_dir, 'train_team1_features.npy'), train_team1)
    np.save(os.path.join(output_dir, 'train_team2_features.npy'), train_team2)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)

    # validation features
    val_team1, val_team2, val_labels = compute_split_features(val_df, "Validation")
    np.save(os.path.join(output_dir, 'val_team1_features.npy'), val_team1)
    np.save(os.path.join(output_dir, 'val_team2_features.npy'), val_team2)
    np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels)

    # test features
    test_team1, test_team2, test_labels = compute_split_features(test_df, "Test")
    np.save(os.path.join(output_dir, 'test_team1_features.npy'), test_team1)
    np.save(os.path.join(output_dir, 'test_team2_features.npy'), test_team2)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)

    metadata = {
        'num_recent_games': num_recent_games,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'feature_dim': 240,
        'stat_columns': stat_columns,
        'random_seed': random_seed
    }

    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n{'='*80}")
    print("Feature Pre-computation Complete!")
    print(f"{'='*80}")
    print(f"\nFeatures saved to: {output_dir}")
    print("\nFiles created:")
    print("  - train_team1_features.npy")
    print("  - train_team2_features.npy")
    print("  - train_labels.npy")
    print("  - val_team1_features.npy")
    print("  - val_team2_features.npy")
    print("  - val_labels.npy")
    print("  - test_team1_features.npy")
    print("  - test_team2_features.npy")
    print("  - test_labels.npy")
    print("  - metadata.pkl")

    print("\nFile sizes:")
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  {filename}: {size_mb:.2f} MB")

    total_size = sum(os.path.getsize(os.path.join(output_dir, f))
                     for f in os.listdir(output_dir))
    print(f"\nTotal size: {total_size / (1024 * 1024):.2f} MB")

    return metadata


if __name__ == "__main__":
    base_dir = "nba_dataset"
    output_dir = "precomputed_features"

    metadata = precompute_all_features(
        team_stats_path=os.path.join(base_dir, "TeamStatistics.csv"),
        season_player_path=os.path.join(base_dir, "Season_player_data.csv"),
        player_stats_path=os.path.join(base_dir, "PlayerStatistics.csv"),
        output_dir=output_dir,
        num_recent_games=5,
        train_ratio=0.7,
        val_ratio=0.15,
        random_seed=42
    )