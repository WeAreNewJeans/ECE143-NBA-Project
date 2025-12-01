import pandas as pd
import numpy as np
from datetime import datetime
import os


def extract_season_from_date(date_str):
    """
    Extract NBA season from game date.
    NBA season spans two calendar years (e.g., 2000-2001 season).
    Season starts in October and ends in June.

    Args:
        date_str: Date string in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        Season string in format 'YYYY-YYYY' (e.g., '2000-2001')
    """
    date = pd.to_datetime(date_str)
    year = date.year
    month = date.month

    if month <= 6:
        season_start = year - 1
        season_end = year
    else:
        season_start = year
        season_end = year + 1

    return f"{season_start}-{season_end}"


def calculate_season_stats(player_games_df):
    """
    Calculate season average statistics for a player.

    Args:
        player_games_df: DataFrame containing all games for one player in one season

    Returns:
        Dictionary containing season statistics
    """
    # Filter out games where player didn't play (numMinutes = 0 or NaN)
    played_games = player_games_df[player_games_df['numMinutes'] > 0].copy()

    if len(played_games) == 0:
        return None

    stats = {
        'games_played': len(played_games),
        'Season_PPG': played_games['points'].mean(),
        'Season_RPG': played_games['reboundsTotal'].mean(),
        'Season_APG': played_games['assists'].mean(),
        'Season_SPG': played_games['steals'].mean(),
        'Season_BPG': played_games['blocks'].mean(),
        'Season_TOV': played_games['turnovers'].mean(),
        'Season_MPG': played_games['numMinutes'].mean(),
    }

    # Field Goal Percentage
    total_fga = played_games['fieldGoalsAttempted'].sum()
    total_fgm = played_games['fieldGoalsMade'].sum()
    stats['Season_FG%'] = total_fgm / total_fga if total_fga > 0 else 0.0

    # Three Point Percentage
    total_3pa = played_games['threePointersAttempted'].sum()
    total_3pm = played_games['threePointersMade'].sum()
    stats['Season_3P%'] = total_3pm / total_3pa if total_3pa > 0 else 0.0

    # Free Throw Percentage
    total_fta = played_games['freeThrowsAttempted'].sum()
    total_ftm = played_games['freeThrowsMade'].sum()
    stats['Season_FT%'] = total_ftm / total_fta if total_fta > 0 else 0.0

    return stats


def preprocess_season_data(input_csv_path, output_csv_path, start_season='2000-2001'):
    """
    Process PlayerStatistics.csv to create season-level aggregated data.

    Args:
        input_csv_path: Path to PlayerStatistics.csv
        output_csv_path: Path to output Season_player_data.csv
        start_season: Starting season to include (default: '2005-2006')
    """
    print(f"Reading player statistics from {input_csv_path}...")
    df = pd.read_csv(input_csv_path, low_memory=False)

    print(f"Total records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    print("\nExtracting seasons from game dates...")
    df['season'] = df['gameDateTimeEst'].apply(extract_season_from_date)

    start_year = int(start_season.split('-')[0])
    df['season_start_year'] = df['season'].apply(lambda x: int(x.split('-')[0]))
    df_filtered = df[df['season_start_year'] >= start_year].copy()

    print(f"Seasons included: {sorted(df_filtered['season'].unique())}")

    print("\nCalculating season statistics for each player...")
    season_data = []

    grouped = df_filtered.groupby(['personId', 'firstName', 'lastName', 'season'])
    total_groups = len(grouped)

    for idx, ((person_id, first_name, last_name, season), group) in enumerate(grouped):
        if (idx + 1) % 1000 == 0:
            print(f"Processing {idx + 1}/{total_groups} player-seasons...")

        stats = calculate_season_stats(group)

        if stats is not None:
            record = {
                'personId': person_id,
                'firstName': first_name,
                'lastName': last_name,
                'season': season,
                **stats
            }
            season_data.append(record)

    season_df = pd.DataFrame(season_data)

    season_df = season_df.sort_values(['season', 'lastName', 'firstName'])

    print(f"\nSaving {len(season_df)} player-season records to {output_csv_path}...")
    season_df.to_csv(output_csv_path, index=False)

    print(f"\nPreprocessing complete!")
    print(f"Total player-seasons: {len(season_df)}")
    print(f"Unique players: {season_df['personId'].nunique()}")
    print(f"Seasons covered: {sorted(season_df['season'].unique())}")

    print("\n" + "="*80)
    print("Sample of processed data:")
    print("="*80)
    print(season_df.head(10).to_string())

    print("\n" + "="*80)
    print("Statistics summary:")
    print("="*80)
    print(season_df.describe())

    return season_df


if __name__ == "__main__":
    base_dir = ""
    input_path = os.path.join(base_dir, "nba_dataset", "PlayerStatistics.csv")
    output_path = os.path.join(base_dir, "nba_dataset", "Season_player_data.csv")

    season_df = preprocess_season_data(
        input_csv_path=input_path,
        output_csv_path=output_path,
        start_season='2000-2001'
    )

    print(f"\nSeason player data saved to: {output_path}")
