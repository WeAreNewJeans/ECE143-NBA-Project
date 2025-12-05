"""
Extract player names and IDs for each game in precomputed features.
This creates a mapping between feature indices and actual player names.
"""

import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm


def extract_season(date_str):
    """Extract NBA season from date string."""
    date = pd.to_datetime(date_str)
    year = date.year
    month = date.month

    if month <= 6:
        return f"{year - 1}-{year}"
    else:
        return f"{year}-{year + 1}"


def get_player_features_for_game(game_id, home_team_name, away_team_name,
                                  player_stats_df, num_players=12):
    """
    Get player information for a specific game.
    Returns player names and IDs in the same order as precomputed features.

    Returns:
        home_players: List of (player_id, player_name) for home team
        away_players: List of (player_id, player_name) for away team
    """
    game_players = player_stats_df[player_stats_df['gameId'] == game_id].copy()

    if len(game_players) == 0:
        return None, None

    home_players = game_players[game_players['playerteamName'] == home_team_name].copy()
    away_players = game_players[game_players['playerteamName'] == away_team_name].copy()

    # Sort by minutes played (descending) to match precompute_features.py logic
    home_players = home_players.sort_values('numMinutes', ascending=False)
    away_players = away_players.sort_values('numMinutes', ascending=False)

    home_players = home_players.head(num_players)
    away_players = away_players.head(num_players)

    # Extract player info
    home_player_info = [(row['personId'], f"{row['firstName']} {row['lastName']}")
                        for _, row in home_players.iterrows()]
    away_player_info = [(row['personId'], f"{row['firstName']} {row['lastName']}")
                        for _, row in away_players.iterrows()]

    # Pad with None if less than num_players
    while len(home_player_info) < num_players:
        home_player_info.append((None, 'UNKNOWN'))
    while len(away_player_info) < num_players:
        away_player_info.append((None, 'UNKNOWN'))

    return home_player_info, away_player_info


def extract_all_player_names(
    team_stats_path,
    player_stats_path,
    output_dir,
    num_players=12
):
    """
    Extract player names for all games in precomputed features.

    Args:
        team_stats_path: Path to TeamStatistics.csv
        player_stats_path: Path to PlayerStatistics.csv
        output_dir: Directory where precomputed features are saved
        num_players: Number of players per team (default: 12)
    """
    print("Now Extracting Player Names from Game Data")

    team_stats_df = pd.read_csv(team_stats_path)
    player_stats_df = pd.read_csv(player_stats_path, low_memory=False)

    print(f"Loaded {len(team_stats_df)} team game records")
    print(f"Loaded {len(player_stats_df)} player game records")

    team_stats_df['season'] = team_stats_df['gameDateTimeEst'].apply(extract_season)
    team_stats_df['season_start_year'] = team_stats_df['season'].apply(
        lambda x: int(x.split('-')[0])
    )
    team_stats_df = team_stats_df[team_stats_df['season_start_year'] >= 2000].copy()

    # Keep only home team records
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

    games_df = games_df.sort_values('gameDateTimeEst').reset_index(drop=True)

    # Exclude ongoing season
    games_df = games_df[games_df['season'] != '2025-2026'].copy()
    print(f"After excluding 2025-2026 season: {len(games_df)} games")

    # Split into train/val/test
    train_val_df = games_df[games_df['season'] < '2024-2025'].copy()
    test_df = games_df[games_df['season'] == '2024-2025'].copy()

    # Further split train_val into train and val
    train_val_df = train_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(len(train_val_df) * 0.9)
    train_df = train_val_df[:train_size].copy()
    val_df = train_val_df[train_size:].copy()

    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} games")
    print(f"  Val:   {len(val_df)} games")
    print(f"  Test:  {len(test_df)} games")

    print("Now extracting Player Information")

    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    player_mappings = {}

    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name} set...")

        split_data = []
        games_without_players = 0

        for idx, game_row in tqdm(split_df.iterrows(), total=len(split_df)):
            game_id = game_row['gameId']
            home_team_id = game_row['teamId']
            home_team_name = game_row['teamName']
            away_team_id = game_row['opponentTeamId']
            away_team_name = game_row['opponentTeamName']

            home_players, away_players = get_player_features_for_game(
                game_id,  home_team_name, away_team_name,
                player_stats_df, num_players
            )

            if home_players is None or away_players is None:
                games_without_players += 1
                home_players = [(None, 'UNKNOWN')] * num_players
                away_players = [(None, 'UNKNOWN')] * num_players

            split_data.append({
                'game_id': game_id,
                'home_team_id': home_team_id,
                'home_team_name': home_team_name,
                'away_team_id': away_team_id,
                'away_team_name': away_team_name,
                'team1_players': home_players,
                'team2_players': away_players,
            })
        player_mappings[split_name] = split_data

        print(f"Games processed: {len(split_data)}")
        print(f"Games without player data: {games_without_players}")

    output_path = os.path.join(output_dir, 'player_names.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(player_mappings, f)

    print(f"\nSaved to: {output_path}")
    
    return player_mappings


def load_player_names(precomputed_dir):
    """
    Load player names from saved file.

    Args:
        precomputed_dir: Directory containing player_names.pkl

    Returns:
        player_mappings: Dictionary with 'train', 'val', 'test' splits
    """
    player_names_path = os.path.join(precomputed_dir, 'player_names.pkl')

    if not os.path.exists(player_names_path):
        raise FileNotFoundError(
            f"Player names file not found at {player_names_path}. "
            "Please run extract_player_names.py first."
        )

    with open(player_names_path, 'rb') as f:
        player_mappings = pickle.load(f)

    return player_mappings


if __name__ == '__main__':
    team_stats_path = 'data/nba_dataset/TeamStatistics.csv'
    player_stats_path = 'data/nba_dataset/PlayerStatistics.csv'
    output_dir = 'data/precomputed_features'

    if not os.path.exists(team_stats_path):
        print(f"ERROR: {team_stats_path} not found!")
        exit(1)

    if not os.path.exists(player_stats_path):
        print(f"ERROR: {player_stats_path} not found!")
        exit(1)

    player_mappings = extract_all_player_names(
        team_stats_path=team_stats_path,
        player_stats_path=player_stats_path,
        output_dir=output_dir,
        num_players=12
    )

    print("Process done. Now you can run analyze scripts.")