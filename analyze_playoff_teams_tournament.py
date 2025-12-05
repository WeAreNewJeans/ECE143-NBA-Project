import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from models.model_transformer import NBAGamePredictionTransformerModel
from extract_player_names import load_player_names

def extract_season_from_date(date_str):
    """Extract NBA season from date string."""
    date = pd.to_datetime(date_str)
    year = date.year
    month = date.month

    if month <= 6:
        return f"{year - 1}-{year}"
    else:
        return f"{year}-{year + 1}"


def identify_playoff_teams(team_stats_path, player_stats_path, season_start_year=2000):
    """
    Identify playoff teams and extract their rosters from last playoff games.

    Returns:
        playoff_teams: List of dicts with team info and player rosters
    """
    team_stats_df = pd.read_csv(team_stats_path)
    player_stats_df = pd.read_csv(player_stats_path, low_memory=False)

    team_stats_df['season'] = team_stats_df['gameDateTimeEst'].apply(extract_season_from_date)
    team_stats_df['gameDateTimeEst'] = pd.to_datetime(
        team_stats_df['gameDateTimeEst'],
        format='ISO8601',
        utc=True
    ).dt.tz_localize(None)

    player_stats_df['gameDateTimeEst'] = pd.to_datetime(
        player_stats_df['gameDateTimeEst'],
        format='ISO8601',
        utc=True
    ).dt.tz_localize(None)

    team_stats_df['season_start_year'] = team_stats_df['season'].apply(
        lambda x: int(x.split('-')[0])
    )
    team_stats_df = team_stats_df[team_stats_df['season_start_year'] >= season_start_year].copy()

    # Identify playoff games (typically April-June)
    team_stats_df['month'] = team_stats_df['gameDateTimeEst'].dt.month
    playoff_games = team_stats_df[team_stats_df['month'].isin([4, 5, 6])].copy()

    print(f"Found {len(playoff_games)} playoff games")

    playoff_teams = []

    for season in playoff_games['season'].unique():
        season_games = playoff_games[playoff_games['season'] == season]

        for team_id in season_games['teamId'].unique():
            team_games = season_games[season_games['teamId'] == team_id].copy()

            if len(team_games) < 4:  # Need at least 4 playoff games
                continue

            # Get last 5 games (most complete roster)
            team_games = team_games.sort_values('gameDateTimeEst', ascending=False)
            last_games = team_games.head(min(5, len(team_games)))

            team_name = last_games.iloc[0]['teamName']

            game_ids = last_games['gameId'].tolist()

            # Find players who played in these games
            team_players = player_stats_df[
                (player_stats_df['gameId'].isin(game_ids)) &
                (player_stats_df['playerteamName'] == team_name)
            ].copy()

            if len(team_players) == 0:
                continue

            player_minutes = team_players.groupby(
                ['personId', 'firstName', 'lastName']
            )['numMinutes'].sum().reset_index()

            player_minutes = player_minutes.sort_values('numMinutes', ascending=False).head(12)

            playoff_teams.append({
                'season': season,
                'team_name': team_name,
                'team_id': team_id,
                'playoff_games': len(team_games),
                'last_game_ids': game_ids,
                'player_roster': [
                    (row['personId'], f"{row['firstName']} {row['lastName']}")
                    for _, row in player_minutes.iterrows()
                ]
            })

    print(f"Identified {len(playoff_teams)} playoff team-season combinations")

    return playoff_teams


def extract_team_features(team_info, precomputed_dir, player_mappings):
    """
    Extract average features for a playoff team from their last playoff games.

    Returns:
        team_features: numpy array of shape (240,) or None if not found
    """
    game_ids = team_info['last_game_ids']
    team_name = team_info['team_name']
    team_features_list = []

    for split in ['train', 'val', 'test']:
        team1 = np.load(os.path.join(precomputed_dir, f'{split}_team1_features.npy'))
        team2 = np.load(os.path.join(precomputed_dir, f'{split}_team2_features.npy'))

        for idx, game_data in enumerate(player_mappings[split]):
            if game_data['game_id'] in game_ids:
                # Check if this team is team1 or team2
                if game_data['home_team_name'] == team_name:
                    features = team1[idx]
                elif game_data['away_team_name'] == team_name:
                    features = team2[idx]
                else:
                    continue

                team_features_list.append(features)

    if len(team_features_list) == 0:
        return None

    # Average features across last few games for robust estimate
    team_features = np.mean(team_features_list, axis=0)
    return team_features


def run_round_robin_tournament(
    model,
    playoff_teams,
    precomputed_dir,
    player_mappings,
    device='cpu'
):
    """
    Simulate round-robin tournament where each team plays every other team.

    Args:
        model: Trained model
        playoff_teams: List of playoff team info
        precomputed_dir: Directory with precomputed features
        player_mappings: Player name mappings
        device: Computing device

    Returns:
        results_df: DataFrame with team rankings based on tournament results
        matchup_matrix: NxN matrix of win probabilities (row team vs column team)
    """
    model.eval()
    model.to(device)

    teams_with_features = []
    for team_info in tqdm(playoff_teams, desc="Loading team features"):
        features = extract_team_features(team_info, precomputed_dir, player_mappings)
        if features is not None:
            teams_with_features.append({
                'team_info': team_info,
                'features': features,
                'team_season_name': f"{team_info['team_name']} ({team_info['season']})"
            })

    n_teams = len(teams_with_features)
    print(f"\nLoaded features for {n_teams} teams")

    print("\n" + "="*80)
    print("Running Round-Robin Tournament")

    team_records = {i: {'wins': 0.0, 'losses': 0.0, 'games': 0} for i in range(n_teams)}
    matchup_matrix = np.zeros((n_teams, n_teams))

    all_features = torch.stack([
        torch.from_numpy(team['features']).float()
        for team in teams_with_features
    ]).to(device)

    # Run tournament: each team vs every other team
    with torch.no_grad():
        for i in tqdm(range(n_teams), desc="Simulating matches"):
            # Team i plays against all other teams
            team_i_features = all_features[i:i+1].repeat(n_teams, 1)
            logits = model(team_i_features, all_features)
            win_probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            matchup_matrix[i, :] = win_probs

            for j in range(n_teams):
                if i != j:
                    win_prob = win_probs[j]
                    team_records[i]['wins'] += win_prob
                    team_records[i]['losses'] += (1 - win_prob)
                    team_records[i]['games'] += 1

    results = []
    for i in range(n_teams):
        team = teams_with_features[i]
        record = team_records[i]

        win_rate = record['wins'] / record['games'] if record['games'] > 0 else 0

        results.append({
            'Rank': 0,
            'Team_Season': team['team_season_name'],
            'Season': team['team_info']['season'],
            'Team_Name': team['team_info']['team_name'],
            'Wins': record['wins'],
            'Losses': record['losses'],
            'Win_Rate': win_rate,
            'Games_Played': record['games'],
            'Playoff_Games': team['team_info']['playoff_games'],
            'Key_Players': ', '.join([name for _, name in team['team_info']['player_roster'][:5]])
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Win_Rate', ascending=False).reset_index(drop=True)
    results_df['Rank'] = range(1, len(results_df) + 1)

    team_names = [team['team_season_name'] for team in teams_with_features]

    return results_df, matchup_matrix, team_names


def visualize_top_teams(results_df, top_n=30, save_path='tournament_top_teams.png'):
    """Visualize top teams by tournament win rate."""
    top_teams = results_df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 10))

    values = top_teams['Win_Rate'].values[::-1]
    colors = plt.cm.RdYlGn(values)

    y_pos = np.arange(top_n)
    ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_teams['Team_Season'].values[::-1], fontsize=9)
    ax.set_xlabel('Win Rate', fontsize=11)
    ax.set_title(f'Top {top_n} Strongest Playoff Teams\n(Round-robin tournament simulation)',
                  fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add win-loss records
    for i, (idx, row) in enumerate(top_teams.iterrows()):
        win_pct = row['Win_Rate'] * 100
        record = f"{row['Wins']:.1f}-{row['Losses']:.1f} ({win_pct:.1f}%)"
        ax.text(row['Win_Rate'] + 0.01, top_n - i - 1,
                 record,
                 va='center', fontsize=7, alpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()



def analyze_specific_matchup(results_df, matchup_matrix, team_names, team1_name, team2_name):
    """Analyze a specific head-to-head matchup."""
    try:
        idx1 = team_names.index(team1_name)
        idx2 = team_names.index(team2_name)

        win_prob_1_vs_2 = matchup_matrix[idx1, idx2]
        win_prob_2_vs_1 = matchup_matrix[idx2, idx1]

        print("\n" + "="*80)
        print(f"Head-to-Head Matchup: {team1_name} vs {team2_name}")
        print("="*80)
        print(f"{team1_name} win probability: {win_prob_1_vs_2*100:.1f}%")
        print(f"{team2_name} win probability: {win_prob_2_vs_1*100:.1f}%")

        # Get team rankings
        rank1 = results_df[results_df['Team_Season'] == team1_name]['Rank'].values[0]
        rank2 = results_df[results_df['Team_Season'] == team2_name]['Rank'].values[0]
        print(f"\nTournament Rankings: #{rank1} vs #{rank2}")

    except ValueError as e:
        print(f"Error: Could not find one or both teams. {e}")


def main():
    parser = argparse.ArgumentParser(description='Analyze playoff teams through tournament simulation')
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
        '--team_stats',
        type=str,
        default='data/nba_dataset/TeamStatistics.csv',
        help='Path to TeamStatistics.csv'
    )
    parser.add_argument(
        '--player_stats',
        type=str,
        default='data/nba_dataset/PlayerStatistics.csv',
        help='Path to PlayerStatistics.csv'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='analysis_results',
        help='Output directory'
    )
    parser.add_argument(
        '--top_n',
        type=int,
        default=50,
        help='Number of top teams to visualize'
    )
    parser.add_argument(
        '--heatmap_n',
        type=int,
        default=20,
        help='Number of teams to show in matchup heatmap'
    )

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print("Playoff Team Tournament Simulation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    model = NBAGamePredictionTransformerModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    player_mappings = load_player_names(args.precomputed_dir)
    playoff_teams = identify_playoff_teams(
        args.team_stats,
        args.player_stats,
        season_start_year=2000
    )

    results_df, matchup_matrix, team_names = run_round_robin_tournament(
        model,
        playoff_teams,
        args.precomputed_dir,
        player_mappings,
        device
    )

    csv_path = os.path.join(args.output_dir, 'tournament_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print(f"Top {args.top_n} Teams")
    print("="*80)
    display_cols = ['Rank', 'Team_Season', 'Win_Rate', 'Wins', 'Losses', 'Key_Players']
    print(results_df[display_cols].head(args.top_n).to_string(index=False))
    print("="*80)
    
    vis_path = os.path.join(args.output_dir, 'tournament_top_teams.png')
    visualize_top_teams(results_df, top_n=args.top_n, save_path=vis_path)

    print("Done!")

if __name__ == '__main__':
    main()
