import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch

from models.model_transformer_with_teaminfo import NBAGamePredictionTransformerModelWithTeamInfo
from dataloaders.dataloader_with_teaminfo import NBAGameDataModuleWithTeamInfo


def train(args):
    """
    Train NBA game prediction model using pre-computed features.
    """
    pl.seed_everything(args.seed, workers=True)

    datamodule = NBAGameDataModuleWithTeamInfo(
        precomputed_dir=args.precomputed_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = NBAGamePredictionTransformerModelWithTeamInfo(
        player_input_dim=args.player_input_dim,
        player_embedding_dim=args.player_embedding_dim,
        num_players=args.num_players,
        team_embedding_dim=args.team_embedding_dim,
        num_heads=args.num_heads,
        num_transformer_layers=args.num_transformer_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="nba-transformer-teaminfo-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
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
            min_delta=0.0001,
            strict=True,
            check_on_train_epoch_end=False,
        )
        callbacks.append(early_stop_callback)
        print(f"Early stopping enabled: patience={args.patience} epochs")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)

    loggers = []
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name="nba_transformer_teaminfo", version=args.experiment_name)
    loggers.append(tb_logger)

    csv_logger = CSVLogger(save_dir=args.log_dir, name="nba_transformer_teaminfo", version=args.experiment_name)
    loggers.append(csv_logger)

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

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    trainer.fit(model, datamodule)

    trainer.test(model, datamodule)

    print("Training Complete")
    print(f"\nBest model: {checkpoint_callback.best_model_path}")
    print(f"Best val loss: {checkpoint_callback.best_model_score:.4f}")

    return model, trainer, checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Train NBA Game Prediction Model with Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data path
    parser.add_argument(
        "--precomputed_dir",
        type=str,
        default="data/precomputed_features_with_teaminfo",
        help="Directory containing pre-computed features",
    )

    # Model hyperparameters
    parser.add_argument("--player_input_dim", type=int, default=20)
    parser.add_argument("--player_embedding_dim", type=int, default=32)
    parser.add_argument("--num_players", type=int, default=12)
    parser.add_argument("--team_embedding_dim", type=int, default=64)

    # Transformer hyperparameters
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Training settings
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed", help="Use mixed precision for faster training")
    parser.add_argument("--deterministic", action="store_true")

    # Callbacks
    parser.add_argument("--early_stopping", action="store_true", default=False, help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="transformer_teaminfo_default")
    parser.add_argument("--log_every_n_steps", type=int, default=50)

    # Other
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Check if pre-computed features exist
    if not os.path.exists(args.precomputed_dir):
        print("ERROR: Pre-computed features not found!")
        print(f"\nDirectory not found: {args.precomputed_dir}")
        print("\nPlease run precompute_features_with_teaminfo.py first")
        exit(1)

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("\nConfiguration:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")

    model, trainer, best_model_path = train(args)

    print("Done!")

if __name__ == "__main__":
    main()
