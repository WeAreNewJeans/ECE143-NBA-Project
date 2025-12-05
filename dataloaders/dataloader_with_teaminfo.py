import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os
import pickle


class PrecomputedNBADatasetWithTeamInfo(Dataset):
    def __init__(self, team1_features, team2_features,
                 team1_home, team2_home,
                 team1_winrate, team2_winrate,
                 team1_season_stats, team2_season_stats,
                 labels):
        """
        Args:
            team1_features: numpy array of shape (N, 240)
            team2_features: numpy array of shape (N, 240)
            team1_home: numpy array of shape (N,)
            team2_home: numpy array of shape (N,)
            team1_winrate: numpy array of shape (N,)
            team2_winrate: numpy array of shape (N,)
            team1_season_stats: numpy array of shape (N, 9)
            team2_season_stats: numpy array of shape (N, 9)
            labels: numpy array of shape (N,)
        """
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
        """
        Returns:
            team1_features: Tensor of shape (240,)
            team2_features: Tensor of shape (240,)
            team1_home: Tensor of shape (1,)
            team2_home: Tensor of shape (1,)
            team1_winrate: Tensor of shape (1,)
            team2_winrate: Tensor of shape (1,)
            team1_season_stats: Tensor of shape (9,)
            team2_season_stats: Tensor of shape (9,)
            label: Tensor of shape (1,)
        """
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
    def __init__(
        self,
        precomputed_dir: str,
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
        self.metadata = None

    def setup(self, stage=None):
        metadata_path = os.path.join(self.precomputed_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        print(f"\nMetadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")

        if stage == "fit" or stage is None:
            train_team1 = np.load(os.path.join(self.precomputed_dir, 'train_team1_features.npy'))
            train_team2 = np.load(os.path.join(self.precomputed_dir, 'train_team2_features.npy'))
            train_team1_home = np.load(os.path.join(self.precomputed_dir, 'train_team1_home.npy'))
            train_team2_home = np.load(os.path.join(self.precomputed_dir, 'train_team2_home.npy'))
            train_team1_winrate = np.load(os.path.join(self.precomputed_dir, 'train_team1_winrate.npy'))
            train_team2_winrate = np.load(os.path.join(self.precomputed_dir, 'train_team2_winrate.npy'))
            train_team1_season_stats = np.load(os.path.join(self.precomputed_dir, 'train_team1_season_stats.npy'))
            train_team2_season_stats = np.load(os.path.join(self.precomputed_dir, 'train_team2_season_stats.npy'))
            train_labels = np.load(os.path.join(self.precomputed_dir, 'train_labels.npy'))

            self.train_dataset = PrecomputedNBADatasetWithTeamInfo(
                train_team1, train_team2,
                train_team1_home, train_team2_home,
                train_team1_winrate, train_team2_winrate,
                train_team1_season_stats, train_team2_season_stats,
                train_labels
            )
            print(f"Training samples: {len(self.train_dataset)}")

            val_team1 = np.load(os.path.join(self.precomputed_dir, 'val_team1_features.npy'))
            val_team2 = np.load(os.path.join(self.precomputed_dir, 'val_team2_features.npy'))
            val_team1_home = np.load(os.path.join(self.precomputed_dir, 'val_team1_home.npy'))
            val_team2_home = np.load(os.path.join(self.precomputed_dir, 'val_team2_home.npy'))
            val_team1_winrate = np.load(os.path.join(self.precomputed_dir, 'val_team1_winrate.npy'))
            val_team2_winrate = np.load(os.path.join(self.precomputed_dir, 'val_team2_winrate.npy'))
            val_team1_season_stats = np.load(os.path.join(self.precomputed_dir, 'val_team1_season_stats.npy'))
            val_team2_season_stats = np.load(os.path.join(self.precomputed_dir, 'val_team2_season_stats.npy'))
            val_labels = np.load(os.path.join(self.precomputed_dir, 'val_labels.npy'))

            self.val_dataset = PrecomputedNBADatasetWithTeamInfo(
                val_team1, val_team2,
                val_team1_home, val_team2_home,
                val_team1_winrate, val_team2_winrate,
                val_team1_season_stats, val_team2_season_stats,
                val_labels
            )
            print(f"Validation samples: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            test_team1 = np.load(os.path.join(self.precomputed_dir, 'test_team1_features.npy'))
            test_team2 = np.load(os.path.join(self.precomputed_dir, 'test_team2_features.npy'))
            test_team1_home = np.load(os.path.join(self.precomputed_dir, 'test_team1_home.npy'))
            test_team2_home = np.load(os.path.join(self.precomputed_dir, 'test_team2_home.npy'))
            test_team1_winrate = np.load(os.path.join(self.precomputed_dir, 'test_team1_winrate.npy'))
            test_team2_winrate = np.load(os.path.join(self.precomputed_dir, 'test_team2_winrate.npy'))
            test_team1_season_stats = np.load(os.path.join(self.precomputed_dir, 'test_team1_season_stats.npy'))
            test_team2_season_stats = np.load(os.path.join(self.precomputed_dir, 'test_team2_season_stats.npy'))
            test_labels = np.load(os.path.join(self.precomputed_dir, 'test_labels.npy'))

            self.test_dataset = PrecomputedNBADatasetWithTeamInfo(
                test_team1, test_team2,
                test_team1_home, test_team2_home,
                test_team1_winrate, test_team2_winrate,
                test_team1_season_stats, test_team2_season_stats,
                test_labels
            )
            print(f"Test samples: {len(self.test_dataset)}")

        print("Data loading complete!")

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
