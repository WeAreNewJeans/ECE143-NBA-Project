import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch

from models.model import NBAGamePredictionModel
from dataloaders.dataloader import NBAGameDataModule


def train(args):
    """
    Train NBA game prediction model using pre-computed features.
    Much faster than computing features on-the-fly!
    """
    # Set random seed
    pl.seed_everything(args.seed, workers=True)

    # Initialize DataModule with pre-computed features
    print("=" * 80)
    print("Initializing DataModule (Pre-computed Features)")
    print("=" * 80)

    datamodule = NBAGameDataModule(
        precomputed_dir=args.precomputed_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Initialize Model
    print("\n" + "=" * 80)
    print("Initializing Model")
    print("=" * 80)

    model = NBAGamePredictionModel(
        player_input_dim=args.player_input_dim,
        player_embedding_dim=args.player_embedding_dim,
        num_players=args.num_players,
        team_embedding_dim=args.team_embedding_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(f"\nModel architecture:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup callbacks
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="nba-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            mode="min",
            verbose=True,
            min_delta=0.0001,  # Minimum change to qualify as improvement
            strict=True,  # Crash if monitored metric is not found
            check_on_train_epoch_end=False,  # Check at validation end
        )
        callbacks.append(early_stop_callback)
        print(f"Early stopping enabled: patience={args.patience} epochs")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)

    # Setup loggers
    loggers = []

    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name="nba_game_prediction", version=args.experiment_name)
    loggers.append(tb_logger)

    csv_logger = CSVLogger(save_dir=args.log_dir, name="nba_game_prediction", version=args.experiment_name)
    loggers.append(csv_logger)

    # Initialize Trainer
    print("\n" + "=" * 80)
    print("Initializing Trainer")
    print("=" * 80)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=args.deterministic,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print(f"\nTraining configuration:")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Accelerator: {args.accelerator}")
    print(f"  Precision: {args.precision}")

    # GPU info
    print("\n" + "=" * 80)
    print("GPU Information")
    print("=" * 80)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Train
    print("\n" + "=" * 80)
    print("Starting Training (Using Pre-computed Features)")
    print("=" * 80)

    trainer.fit(model, datamodule)

    # Test
    print("\n" + "=" * 80)
    print("Testing Model")
    print("=" * 80)

    trainer.test(model, datamodule)

    # Summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest model: {checkpoint_callback.best_model_path}")
    print(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")

    return model, trainer, checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Train NBA Game Prediction Model (Pre-computed Features)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data path
    parser.add_argument(
        "--precomputed_dir",
        type=str,
        default="data/precomputed_features",
        help="Directory containing pre-computed features",
    )

    # Model hyperparameters (Lightweight architecture)
    parser.add_argument("--player_input_dim", type=int, default=20)
    parser.add_argument("--player_embedding_dim", type=int, default=16, help="Player embedding dimension (20->32->16)")
    parser.add_argument("--num_players", type=int, default=12)
    parser.add_argument("--team_embedding_dim", type=int, default=32, help="Team embedding dimension (192->64->32)")

    # Training hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size (can be larger with pre-computed features)"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Training settings
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed", help="Use mixed precision for faster training")
    parser.add_argument("--deterministic", action="store_true")

    # Callbacks
    parser.add_argument(
        "--early_stopping", action="store_true", default=False, help="Enable early stopping (default: True)"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience - stop if no improvement for N epochs"
    )

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--log_every_n_steps", type=int, default=50)

    # Other
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Check if pre-computed features exist
    if not os.path.exists(args.precomputed_dir):
        print("=" * 80)
        print("ERROR: Pre-computed features not found!")
        print("=" * 80)
        print(f"\nDirectory not found: {args.precomputed_dir}")
        print("\nPlease run the following command first:")
        print("  python precompute_features.py")
        print("\nThis will pre-compute all features and save them to disk.")
        exit(1)

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 80)
    print("NBA Game Prediction Model Training (Pre-computed Features)")
    print("=" * 80)
    print("\nConfiguration:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")

    # Train
    model, trainer, best_model_path = train(args)

    print("\n" + "=" * 80)
    print("All Done!")
    print("=" * 80)
    print(f"\nTo view logs, run:")
    print(f"  tensorboard --logdir {args.log_dir}")
    print(f"\nBest model: {best_model_path}")


if __name__ == "__main__":
    main()
