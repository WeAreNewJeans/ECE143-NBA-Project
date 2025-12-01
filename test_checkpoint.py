"""
Test a trained model checkpoint on the test set.
Supports both base transformer and transformer with team info.
"""
import os
import argparse
import pytorch_lightning as pl
import torch

from model_transformer_with_teaminfo import NBAGamePredictionTransformerModelWithTeamInfo as TeamInfoModel
from dataloader_with_teaminfo import NBAGameDataModuleWithTeamInfo


def test_checkpoint(checkpoint_path, batch_size=64, num_workers=0):
    """
    Test a checkpoint (with team info) on the test set.

    Args:
        checkpoint_path: Path to the checkpoint file (.ckpt)
        batch_size: Batch size for testing
        num_workers: Number of workers for data loading
    """
    print("="*80)
    print("Testing Model from Checkpoint (with Team Info)")
    print("="*80)
    print(f"\nCheckpoint: {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print("\n" + "="*80)
        print("ERROR: Checkpoint not found!")
        print("="*80)
        print(f"\nCheckpoint path: {checkpoint_path}")
        exit(1)

    # Setup data
    precomputed_dir = r'c:\Users\jeffc\Desktop\ECE 143\project\data\precomputed_features_with_teaminfo'

    print("\n" + "="*80)
    print("Loading DataModule with Team Info")
    print("="*80)

    # Check if precomputed features exist
    if not os.path.exists(precomputed_dir):
        print("\n" + "="*80)
        print("ERROR: Pre-computed features not found!")
        print("="*80)
        print(f"\nDirectory: {precomputed_dir}")
        exit(1)

    datamodule = NBAGameDataModuleWithTeamInfo(
        precomputed_dir=precomputed_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Setup datamodule to load data
    print("\nSetting up datamodule...")
    datamodule.setup()

    # Load model
    print("\n" + "="*80)
    print("Loading Model with Team Info")
    print("="*80)
    model = TeamInfoModel.load_from_checkpoint(checkpoint_path)

    # Setup trainer
    print("\n" + "="*80)
    print("Initializing Trainer")
    print("="*80)

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Print model info
    print(f"\nModel loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # GPU info
    print("\n" + "="*80)
    print("GPU Information")
    print("="*80)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Test on validation set
    print("\n" + "="*80)
    print("Testing on Validation Set")
    print("="*80)

    val_results = trainer.test(model, datamodule.val_dataloader())

    # Test on test set
    print("\n" + "="*80)
    print("Testing on Test Set")
    print("="*80)

    test_results = trainer.test(model, datamodule.test_dataloader())

    # Summary
    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80)

    print("\nValidation Set:")
    for key, value in val_results[0].items():
        print(f"  {key}: {value:.4f}")

    print("\nTest Set:")
    for key, value in test_results[0].items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)

    return val_results, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test a trained model checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (.ckpt)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for testing'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of workers for data loading'
    )

    args = parser.parse_args()

    # Test checkpoint
    val_results, test_results = test_checkpoint(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
