import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pickle


class PrecomputedNBADatasetWithTeamInfo(Dataset):
    """
    PyTorch Dataset for pre-computed NBA game features with team information.

    Returns:
        team1_features: (240,) player features for team 1
        team2_features: (240,) player features for team 2
        team1_home: (1,) is team 1 home? (1.0=home, 0.0=away)
        team2_home: (1,) is team 2 home? (1.0=home, 0.0=away)
        team1_winrate: (1,) team 1 win rate before this game
        team2_winrate: (1,) team 2 win rate before this game
        team1_season_stats: (9,) team 1 season stats [PPG, RPG, APG, SPG, BPG, TOV, FG%, 3P%, FT%]
        team2_season_stats: (9,) team 2 season stats [PPG, RPG, APG, SPG, BPG, TOV, FG%, 3P%, FT%]
        label: (1,) did team 1 win?
    """

    def __init__(self, team1_features, team2_features,
                 team1_home, team2_home,
                 team1_winrate, team2_winrate,
                 team1_season_stats, team2_season_stats,
                 labels):
        self.team1_features = team1_features
        self.team2_features = team2_features
        self.team1_home = team1_home
        self.team2_home = team2_home
        self.team1_winrate = team1_winrate
        self.team2_winrate = team2_winrate
        self.team1_season_stats = team1_season_stats
        self.team2_season_stats = team2_season_stats
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.team1_features[idx]).float(),
            torch.from_numpy(self.team2_features[idx]).float(),
            torch.tensor([self.team1_home[idx]], dtype=torch.float32),
            torch.tensor([self.team2_home[idx]], dtype=torch.float32),
            torch.tensor([self.team1_winrate[idx]], dtype=torch.float32),
            torch.tensor([self.team2_winrate[idx]], dtype=torch.float32),
            torch.from_numpy(self.team1_season_stats[idx]).float(),
            torch.from_numpy(self.team2_season_stats[idx]).float(),
            torch.tensor([self.labels[idx]], dtype=torch.float32)
        )


class NBAGameDataModuleWithTeamInfo(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for NBA game prediction with team information.
    Loads pre-computed features from disk.
    """

    def __init__(
        self,
        precomputed_dir: str = 'data/precomputed_features_with_teaminfo',
        batch_size: int = 32,
        num_workers: int = 4
    ):
        super().__init__()
        self.precomputed_dir = precomputed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Load pre-computed features from disk."""
        print(f"\nLoading pre-computed features from: {self.precomputed_dir}")

        # Load training data
        train_team1 = np.load(os.path.join(self.precomputed_dir, 'train_team1_features.npy'))
        train_team2 = np.load(os.path.join(self.precomputed_dir, 'train_team2_features.npy'))
        train_team1_home = np.load(os.path.join(self.precomputed_dir, 'train_team1_home.npy'))
        train_team2_home = np.load(os.path.join(self.precomputed_dir, 'train_team2_home.npy'))
        train_team1_winrate = np.load(os.path.join(self.precomputed_dir, 'train_team1_winrate.npy'))
        train_team2_winrate = np.load(os.path.join(self.precomputed_dir, 'train_team2_winrate.npy'))
        train_team1_season_stats = np.load(os.path.join(self.precomputed_dir, 'train_team1_season_stats.npy'))
        train_team2_season_stats = np.load(os.path.join(self.precomputed_dir, 'train_team2_season_stats.npy'))
        train_labels = np.load(os.path.join(self.precomputed_dir, 'train_labels.npy'))

        # Load validation data
        val_team1 = np.load(os.path.join(self.precomputed_dir, 'val_team1_features.npy'))
        val_team2 = np.load(os.path.join(self.precomputed_dir, 'val_team2_features.npy'))
        val_team1_home = np.load(os.path.join(self.precomputed_dir, 'val_team1_home.npy'))
        val_team2_home = np.load(os.path.join(self.precomputed_dir, 'val_team2_home.npy'))
        val_team1_winrate = np.load(os.path.join(self.precomputed_dir, 'val_team1_winrate.npy'))
        val_team2_winrate = np.load(os.path.join(self.precomputed_dir, 'val_team2_winrate.npy'))
        val_team1_season_stats = np.load(os.path.join(self.precomputed_dir, 'val_team1_season_stats.npy'))
        val_team2_season_stats = np.load(os.path.join(self.precomputed_dir, 'val_team2_season_stats.npy'))
        val_labels = np.load(os.path.join(self.precomputed_dir, 'val_labels.npy'))

        # Load test data
        test_team1 = np.load(os.path.join(self.precomputed_dir, 'test_team1_features.npy'))
        test_team2 = np.load(os.path.join(self.precomputed_dir, 'test_team2_features.npy'))
        test_team1_home = np.load(os.path.join(self.precomputed_dir, 'test_team1_home.npy'))
        test_team2_home = np.load(os.path.join(self.precomputed_dir, 'test_team2_home.npy'))
        test_team1_winrate = np.load(os.path.join(self.precomputed_dir, 'test_team1_winrate.npy'))
        test_team2_winrate = np.load(os.path.join(self.precomputed_dir, 'test_team2_winrate.npy'))
        test_team1_season_stats = np.load(os.path.join(self.precomputed_dir, 'test_team1_season_stats.npy'))
        test_team2_season_stats = np.load(os.path.join(self.precomputed_dir, 'test_team2_season_stats.npy'))
        test_labels = np.load(os.path.join(self.precomputed_dir, 'test_labels.npy'))

        # Load metadata
        with open(os.path.join(self.precomputed_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        print(f"\nDataset sizes:")
        print(f"  Training: {len(train_labels)} games")
        print(f"  Validation: {len(val_labels)} games")
        print(f"  Test: {len(test_labels)} games")

        print(f"\nFeature dimensions:")
        print(f"  Player features: {train_team1.shape[1]}-dim (12 players × 20 features)")
        print(f"  Team info: 11-dim (home/away + win rate + 9 season stats)")

        # Create datasets
        self.train_dataset = PrecomputedNBADatasetWithTeamInfo(
            train_team1, train_team2,
            train_team1_home, train_team2_home,
            train_team1_winrate, train_team2_winrate,
            train_team1_season_stats, train_team2_season_stats,
            train_labels
        )

        self.val_dataset = PrecomputedNBADatasetWithTeamInfo(
            val_team1, val_team2,
            val_team1_home, val_team2_home,
            val_team1_winrate, val_team2_winrate,
            val_team1_season_stats, val_team2_season_stats,
            val_labels
        )

        self.test_dataset = PrecomputedNBADatasetWithTeamInfo(
            test_team1, test_team2,
            test_team1_home, test_team2_home,
            test_team1_winrate, test_team2_winrate,
            test_team1_season_stats, test_team2_season_stats,
            test_labels
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )


if __name__ == "__main__":
    # Test datamodule
    print("Testing NBAGameDataModuleWithTeamInfo...")

    datamodule = NBAGameDataModuleWithTeamInfo(
        precomputed_dir='data/precomputed_features_with_teaminfo',
        batch_size=16,
        num_workers=0
    )

    datamodule.setup()

    # Test train dataloader
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    (team1_features, team2_features, team1_home, team2_home, team1_winrate, team2_winrate,
     team1_season_stats, team2_season_stats, labels) = batch

    print(f"\nBatch shapes:")
    print(f"  team1_features: {team1_features.shape}")
    print(f"  team2_features: {team2_features.shape}")
    print(f"  team1_home: {team1_home.shape}")
    print(f"  team2_home: {team2_home.shape}")
    print(f"  team1_winrate: {team1_winrate.shape}")
    print(f"  team2_winrate: {team2_winrate.shape}")
    print(f"  team1_season_stats: {team1_season_stats.shape}")
    print(f"  team2_season_stats: {team2_season_stats.shape}")
    print(f"  labels: {labels.shape}")

    print(f"\nSample data:")
    print(f"  Team1 is home: {team1_home[0].item()}")
    print(f"  Team1 win rate: {team1_winrate[0].item():.3f}")
    print(f"  Team1 season stats (PPG, RPG, APG...): {team1_season_stats[0].tolist()}")
    print(f"  Team2 is home: {team2_home[0].item()}")
    print(f"  Team2 win rate: {team2_winrate[0].item():.3f}")
    print(f"  Team2 season stats (PPG, RPG, APG...): {team2_season_stats[0].tolist()}")
    print(f"  Label (team1 wins): {labels[0].item()}")

    print("\nDataModule test passed!")
