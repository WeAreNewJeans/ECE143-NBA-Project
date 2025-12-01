import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional
import os
import pickle


class PrecomputedNBADataset(Dataset):
    """
    Dataset using pre-computed features.
    Much faster than computing features on-the-fly.
    """
    def __init__(self, team1_features, team2_features, labels):
        """
        Args:
            team1_features: numpy array of shape (N, 240)
            team2_features: numpy array of shape (N, 240)
            labels: numpy array of shape (N,)
        """
        self.team1_features = team1_features
        self.team2_features = team2_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns:
            team1_features: Tensor of shape (240,)
            team2_features: Tensor of shape (240,)
            label: Tensor of shape (1,)
        """
        return (
            torch.from_numpy(self.team1_features[idx]).float(),
            torch.from_numpy(self.team2_features[idx]).float(),
            torch.tensor([self.labels[idx]], dtype=torch.float32)
        )


class NBAGameDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule using pre-computed features.
    """
    def __init__(
        self,
        precomputed_dir: str,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Args:
            precomputed_dir: Directory containing pre-computed .npy files
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
        """
        super().__init__()
        self.precomputed_dir = precomputed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.metadata = None

    def setup(self, stage: Optional[str] = None):
        """Load pre-computed features."""
        print("="*80)
        print("Loading Pre-computed Features")
        print("="*80)

        # Load metadata
        metadata_path = os.path.join(self.precomputed_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        print(f"\nMetadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")

        # Load training data
        if stage == "fit" or stage is None:
            print("\nLoading training data...")
            train_team1 = np.load(os.path.join(self.precomputed_dir, 'train_team1_features.npy'))
            train_team2 = np.load(os.path.join(self.precomputed_dir, 'train_team2_features.npy'))
            train_labels = np.load(os.path.join(self.precomputed_dir, 'train_labels.npy'))

            self.train_dataset = PrecomputedNBADataset(train_team1, train_team2, train_labels)
            print(f"  Training samples: {len(self.train_dataset)}")

            print("\nLoading validation data...")
            val_team1 = np.load(os.path.join(self.precomputed_dir, 'val_team1_features.npy'))
            val_team2 = np.load(os.path.join(self.precomputed_dir, 'val_team2_features.npy'))
            val_labels = np.load(os.path.join(self.precomputed_dir, 'val_labels.npy'))

            self.val_dataset = PrecomputedNBADataset(val_team1, val_team2, val_labels)
            print(f"  Validation samples: {len(self.val_dataset)}")

        # Load test data
        if stage == "test" or stage is None:
            print("\nLoading test data...")
            test_team1 = np.load(os.path.join(self.precomputed_dir, 'test_team1_features.npy'))
            test_team2 = np.load(os.path.join(self.precomputed_dir, 'test_team2_features.npy'))
            test_labels = np.load(os.path.join(self.precomputed_dir, 'test_labels.npy'))

            self.test_dataset = PrecomputedNBADataset(test_team1, test_team2, test_labels)
            print(f"  Test samples: {len(self.test_dataset)}")

        print("\n" + "="*80)
        print("Data loading complete! (Much faster than on-the-fly computation)")
        print("="*80)

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


# Usage example
if __name__ == "__main__":
    precomputed_dir = r"c:\Users\jeffc\Desktop\ECE 143\project\data\precomputed_features"

    # Check if pre-computed features exist
    if not os.path.exists(precomputed_dir):
        print(f"Error: Pre-computed features not found at {precomputed_dir}")
        print("\nPlease run precompute_features.py first:")
        print("  python precompute_features.py")
        exit(1)

    # Initialize DataModule
    print("Initializing DataModule with pre-computed features...")
    datamodule = NBAGameDataModule(
        precomputed_dir=precomputed_dir,
        batch_size=64,
        num_workers=4
    )

    # Setup data
    datamodule.setup()

    # Test dataloader speed
    print("\n" + "="*80)
    print("Testing DataLoader Speed")
    print("="*80)

    import time

    train_loader = datamodule.train_dataloader()

    print(f"\nIterating through training data...")
    start_time = time.time()

    for batch_idx, (team1_features, team2_features, labels) in enumerate(train_loader):
        if batch_idx == 0:
            print(f"  Batch {batch_idx + 1}:")
            print(f"    Team 1 features shape: {team1_features.shape}")
            print(f"    Team 2 features shape: {team2_features.shape}")
            print(f"    Labels shape: {labels.shape}")

        if batch_idx >= 99:  # Test first 100 batches
            break

    elapsed = time.time() - start_time
    print(f"\nTime to load 100 batches: {elapsed:.2f} seconds")
    print(f"Average time per batch: {elapsed/100*1000:.2f} ms")

    print("\n" + "="*80)
    print("Pre-computed features are MUCH faster!")
    print("="*80)
