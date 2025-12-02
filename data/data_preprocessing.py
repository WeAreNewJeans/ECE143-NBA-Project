import pandas as pd
import numpy as np
from datetime import datetime
import os

def extract_season_from_date(date_str):
    """
    Extract NBA season from game date.
    NBA season spans two calendar years (e.g., 2005-2006 season).
    Season starts in October and ends in June.

    Args:
        date_str: Date string in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        Season string in format 'YYYY-YYYY' (e.g., '2005-2006')
    """
    date = pd.to_datetime(date_str)
    year = date.year
    month = date.month

    # If month is January-June, season started previous year
    # If month is July-December, season starts current year
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
        Dictionary containing season statistics, or None if player never played.
    """
    # Ensure numMinutes is numeric
    player_games_df = player_games_df.copy()
    player_games_df["numMinutes"] = pd.to_numeric(
        player_games_df["numMinutes"], errors="coerce"
    ).fillna(0)

    # Filter out games where player didn't play (numMinutes = 0 or NaN)
    played_games = player_games_df[player_games_df["numMinutes"] > 0]

    if len(played_games) == 0:
        return None

    # Calculate per-game averages
    stats = {
        "games_played": len(played_games),
        "Season_PPG": played_games["points"].mean(),
        "Season_RPG": played_games["reboundsTotal"].mean(),
        "Season_APG": played_games["assists"].mean(),
        "Season_SPG": played_games["steals"].mean(),
        "Season_BPG": played_games["blocks"].mean(),
        "Season_TOV": played_games["turnovers"].mean(),
        "Season_MPG": played_games["numMinutes"].mean(),
    }

    # Calculate shooting percentages (weighted by attempts)

    # Field Goal Percentage
    total_fga = played_games["fieldGoalsAttempted"].sum()
    total_fgm = played_games["fieldGoalsMade"].sum()
    stats["Season_FG%"] = total_fgm / total_fga if total_fga > 0 else 0.0

    # Three Point Percentage
    total_3pa = played_games["threePointersAttempted"].sum()
    total_3pm = played_games["threePointersMade"].sum()
    stats["Season_3P%"] = total_3pm / total_3pa if total_3pa > 0 else 0.0

    # Free Throw Percentage
    total_fta = played_games["freeThrowsAttempted"].sum()
    total_ftm = played_games["freeThrowsMade"].sum()
    stats["Season_FT%"] = total_ftm / total_fta if total_fta > 0 else 0.0

    return stats


def preprocess_season_data(input_csv_path, output_csv_path, start_season="2005-2006"):
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

    # Extract season from game date
    print("\nExtracting seasons from game dates...")
    df["season"] = df["gameDateTimeEst"].apply(extract_season_from_date)

    # Filter seasons from start_season onwards
    start_year = int(start_season.split("-")[0])
    df["season_start_year"] = df["season"].apply(lambda x: int(x.split("-")[0]))
    df_filtered = df[df["season_start_year"] >= start_year].copy()

    print(
        f"\nFiltered to seasons from {start_season} onwards: {len(df_filtered)} records"
    )
    print(f"Seasons included: {sorted(df_filtered['season'].unique())}")

    # Ensure numeric for stats columns we will use
    numeric_cols = [
        "numMinutes",
        "points",
        "reboundsTotal",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "fieldGoalsAttempted",
        "fieldGoalsMade",
        "threePointersAttempted",
        "threePointersMade",
        "freeThrowsAttempted",
        "freeThrowsMade",
    ]
    for col in numeric_cols:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce").fillna(
                0
            )

    # Group by player and season (personId, season) – use names only for display
    print("\nCalculating season statistics for each player...")
    season_data = []

    grouped = df_filtered.groupby(["personId", "season"])
    total_groups = len(grouped)

    for idx, ((person_id, season), group) in enumerate(grouped):
        if (idx + 1) % 1000 == 0:
            print(f"Processing {idx + 1}/{total_groups} player-seasons...")

        stats = calculate_season_stats(group)

        if stats is not None:
            # Take first occurrence of name for that player-season
            first_row = group.iloc[0]
            record = {
                "personId": person_id,
                "firstName": first_row.get("firstName", ""),
                "lastName": first_row.get("lastName", ""),
                "season": season,
                **stats,
            }
            season_data.append(record)

    # Create DataFrame
    season_df = pd.DataFrame(season_data)

    # Sort by season and player name
    if not season_df.empty:
        season_df = season_df.sort_values(["season", "lastName", "firstName"])

    # Save to CSV
    print(f"\nSaving {len(season_df)} player-season records to {output_csv_path}...")
    season_df.to_csv(output_csv_path, index=False)

    print("\nPreprocessing complete!")
    print(f"Total player-seasons: {len(season_df)}")
    print(f"Unique players: {season_df['personId'].nunique()}")
    print(f"Seasons covered: {sorted(season_df['season'].unique())}")

    print("\n" + "=" * 80)
    print("Sample of processed data:")
    print("=" * 80)
    print(season_df.head(10).to_string())

    print("\n" + "=" * 80)
    print("Statistics summary:")
    print("=" * 80)
    print(season_df.describe())

    return season_df


def prepare_team_game_stats(input_path, output_path):
    input_path = "/datasets/PlayerStatistics.csv"
    output_path = "/datasets/Season_player_data.csv"

    season_df = preprocess_season_data(
        input_csv_path=input_path,
        output_csv_path=output_path,
        start_season='2005-2006'
    )

    SAVE_PATH_CSV = "/datasets/team_game_stats.csv"
    team_game_stats.to_csv(SAVE_PATH_CSV, index=False)
    print("Saved CSV to:", SAVE_PATH_CSV)

def prepare_model_data(team_game_stats):
    
    GAMES_PATH = "/datasets/Games.csv"

    games_df = pd.read_csv(GAMES_PATH, low_memory=False)

    print("Games DF shape:", games_df.shape)
    print("\nColumns:")
    print(games_df.columns.tolist())

    print("\nHead:")
    print(games_df.head())

    games_df = pd.read_csv("/datasets/Games.csv", low_memory=False)

    games_df["season"] = games_df["gameDateTimeEst"].apply(extract_season_from_date)

    games_df["season_start_year"] = games_df["season"].str.split("-").str[0].astype(int)
    games_df = games_df[games_df["season_start_year"] >= 2005].copy()

    print("Games after season filtering:", games_df.shape)

    #ceate home_win label 
    games_df["home_win"] = (games_df["homeScore"] > games_df["awayScore"]).astype(int)

    print("\nHome win distribution:")
    print(games_df["home_win"].value_counts(normalize=True))

    games_df.to_parquet("/datasets/games_cleaned.parquet", index=False)
    print("\nSaved cleaned games to /datasets/games_cleaned.parquet")

    # Preview
    games_df.head()

    # Load cleaned games table
    games_df = pd.read_parquet("/datasets/games_cleaned.parquet")
    print("games_df shape:", games_df.shape)

    # Basic sanity on team_game_stats
    print("team_game_stats shape:", team_game_stats.shape)
    print(team_game_stats.head())

    # Make sure key columns line up and are clean
    team_game_stats["playerteamName"] = team_game_stats["playerteamName"].astype(str).str.strip()
    games_df["hometeamName"] = games_df["hometeamName"].astype(str).str.strip()
    games_df["awayteamName"] = games_df["awayteamName"].astype(str).str.strip()

    # Season feature columns (same as before)
    season_features_cols = [
        "Season_PPG", "Season_RPG", "Season_APG", "Season_SPG", "Season_BPG",
        "Season_TOV", "Season_MPG", "Season_FG%", "Season_3P%", "Season_FT%"
    ]

    # Merge HOME team season stats
    home_stats = team_game_stats.rename(columns={"playerteamName": "hometeamName"})
    games_with_home = games_df.merge(
        home_stats[["gameId", "hometeamName"] + season_features_cols],
        on=["gameId", "hometeamName"],
        how="inner"
    )

    home_rename = {c: f"home_{c}" for c in season_features_cols}
    games_with_home = games_with_home.rename(columns=home_rename)

    print("After home merge:", games_with_home.shape)

    # Merge AWAY team season stats
    away_stats = team_game_stats.rename(columns={"playerteamName": "awayteamName"})
    games_with_both = games_with_home.merge(
        away_stats[["gameId", "awayteamName"] + season_features_cols],
        on=["gameId", "awayteamName"],
        how="inner"
    )

    away_rename = {c: f"away_{c}" for c in season_features_cols}
    games_with_both = games_with_both.rename(columns=away_rename)

    print("After away merge (games_with_both):", games_with_both.shape)

    #Check NA ratios in features
    feature_cols = [c for c in games_with_both.columns 
                    if c.startswith("home_Season_") or c.startswith("away_Season_")]

    print("\nNA ratio in model features:")
    print(games_with_both[feature_cols].isna().mean())

    #Drop rows with missing features (if any)
    games_model_df = games_with_both.dropna(subset=feature_cols).copy()
    print("\nFinal games_model_df shape:", games_model_df.shape)

    # Quick preview
    print(games_model_df[[
        "gameId", "hometeamName", "awayteamName", "homeScore", "awayScore", "home_win"
    ]].head())

    #Save for later blocks
    games_model_df.to_parquet("/datasets/games_model_df.parquet", index=False)
    print("\nSaved games_model_df to /datasets/games_model_df.parquet")


if __name__ == "__main__":
    prepare_team_game_stats(
        input_path="/datasets/PlayerStatistics.csv",
        output_path="/datasets/Season_player_data.csv"
    )

    team_game_stats = pd.read_csv("/datasets/team_game_stats.csv")

    prepare_model_data(team_game_stats)