import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math


class PlayerEncoder(nn.Module):
    """
    Encodes individual player feature vectors
    Input: 20-dim (10 season avg features + 10 recent 5-game avg features)
    Output: 32-dim embedding (increased for transformer)
    """

    def __init__(self, input_dim=20, hidden_dim=48, output_dim=32):
        super(PlayerEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, output_dim), nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, 20) - player feature vector
        Returns:
            shape (batch_size, output_dim) - encoded player vector
        """
        return self.encoder(x)


class PositionalEncoding(nn.Module):
    """
    Adds positional information to player embeddings.

    Note: Players are sorted by playing time in precompute_features.py
    """

    def __init__(self, d_model, max_len=12):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix using sinusoidal functions
        # This allows the model to distinguish player order/importance
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, seq_len, d_model)
        Returns:
            shape (batch_size, seq_len, d_model) - with positional encoding added
        """
        return x + self.pe[: x.size(1), :].unsqueeze(0)


class TransformerTeamEncoder(nn.Module):
    """
    Uses Transformer to aggregate player embeddings into team representation
    This captures player interactions and teamwork dynamics

    Input: 240-dim (12 players * 20-dim features)
    Output: 64-dim team embedding

    Architecture:
    1. Encode each player: 20 -> 48 -> 32
    2. Add positional encoding to distinguish player order
    3. Multi-head self-attention to capture player interactions
    4. Feed-forward network for final team representation
    """

    def __init__(
        self,
        player_input_dim=20,
        player_embedding_dim=32,
        num_players=12,
        team_output_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
    ):
        super(TransformerTeamEncoder, self).__init__()
        self.num_players = num_players
        self.player_input_dim = player_input_dim
        self.player_embedding_dim = player_embedding_dim

        # Player encoder: 20 -> 48 -> 32
        self.player_encoder = PlayerEncoder(input_dim=player_input_dim, hidden_dim=48, output_dim=player_embedding_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=player_embedding_dim, max_len=num_players)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=player_embedding_dim,
            nhead=num_heads,
            dim_feedforward=player_embedding_dim * 2,  # 64
            dropout=dropout,
            activation="relu",
            batch_first=True,  # Important: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Team aggregation: Pool transformer output and project to team embedding
        # We'll use mean pooling over all players
        self.team_projection = nn.Sequential(
            nn.Linear(player_embedding_dim, team_output_dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, 240) - concatenated features of 12 players
        Returns:
            shape (batch_size, team_output_dim) - team vector
        """
        batch_size = x.size(0)

        # Reshape to (batch_size, 12, 20)
        x = x.view(batch_size, self.num_players, self.player_input_dim)

        # Encode each player: (batch_size, 12, 20) -> (batch_size, 12, 32)
        # Reshape to (batch_size * 12, 20)
        x = x.view(batch_size * self.num_players, self.player_input_dim)
        player_embeddings = self.player_encoder(x)  # (batch_size * 12, 32)

        # Reshape back to (batch_size, 12, 32)
        player_embeddings = player_embeddings.view(batch_size, self.num_players, self.player_embedding_dim)

        # Add positional encoding
        player_embeddings = self.positional_encoding(player_embeddings)

        # Apply transformer encoder to capture player interactions
        # Output: (batch_size, 12, 32)
        transformer_output = self.transformer_encoder(player_embeddings)

        # Aggregate all players using mean pooling
        # (batch_size, 12, 32) -> (batch_size, 32)
        team_representation = transformer_output.mean(dim=1)

        # Project to final team embedding
        # (batch_size, 32) -> (batch_size, team_output_dim)
        team_embedding = self.team_projection(team_representation)

        return team_embedding


class GamePredictor(nn.Module):
    """
    Takes two team vectors and predicts the win probability of the first team
    Architecture: 128 (2*64) -> 64 -> 32 -> 1
    """

    def __init__(self, team_embedding_dim=64):
        super(GamePredictor, self).__init__()

        # Concatenate two team vectors and make prediction
        # Note: No Sigmoid here - we use BCEWithLogitsLoss which includes sigmoid
        self.predictor = nn.Sequential(
            nn.Linear(team_embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),  # Output logits (not probabilities)
        )

    def forward(self, team1_embedding, team2_embedding):
        """
        Args:
            team1_embedding: shape (batch_size, team_embedding_dim) - first team vector
            team2_embedding: shape (batch_size, team_embedding_dim) - second team vector
        Returns:
            shape (batch_size, 1) - win logits of the first team
        """
        # Concatenate two team vectors
        combined = torch.cat([team1_embedding, team2_embedding], dim=1)
        win_logits = self.predictor(combined)
        return win_logits


class NBAGamePredictionTransformerModel(pl.LightningModule):
    """
    NBA game prediction model using Transformer for team encoding
    """

    def __init__(
        self,
        player_input_dim=20,
        player_embedding_dim=32,
        num_players=12,
        team_embedding_dim=64,
        num_heads=4,
        num_transformer_layers=2,
        dropout=0.3,
        learning_rate=1e-3,
        weight_decay=1e-4,
    ):
        super(NBAGamePredictionTransformerModel, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Transformer-based team encoder
        self.team_encoder = TransformerTeamEncoder(
            player_input_dim=player_input_dim,
            player_embedding_dim=player_embedding_dim,
            num_players=num_players,
            team_output_dim=team_embedding_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
        )

        # Game predictor
        self.game_predictor = GamePredictor(team_embedding_dim=team_embedding_dim)

        # Loss function - BCEWithLogitsLoss is safe for mixed precision training
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, team1_features, team2_features):
        """
        Args:
            team1_features: shape (batch_size, 240) - features of the first team
            team2_features: shape (batch_size, 240) - features of the second team
        Returns:
            shape (batch_size, 1) - win logits of the first team
        """
        # Encode both teams using transformer
        team1_embedding = self.team_encoder(team1_features)
        team2_embedding = self.team_encoder(team2_features)

        # Predict win probability
        win_logits = self.game_predictor(team1_embedding, team2_embedding)

        return win_logits

    def training_step(self, batch, batch_idx):
        """Training step"""
        team1_features, team2_features, labels = batch

        # Forward pass (returns logits)
        logits = self(team1_features, team2_features)

        # Compute loss (BCEWithLogitsLoss expects logits and labels)
        loss = self.criterion(logits.squeeze(), labels.squeeze().float())

        # Compute accuracy (convert logits to probabilities with sigmoid)
        probabilities = torch.sigmoid(logits.squeeze())
        predicted_labels = (probabilities > 0.5).float()
        accuracy = (predicted_labels == labels.squeeze()).float().mean()

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        team1_features, team2_features, labels = batch

        # Forward pass (returns logits)
        logits = self(team1_features, team2_features)

        # Compute loss
        loss = self.criterion(logits.squeeze(), labels.squeeze().float())

        # Compute accuracy (convert logits to probabilities with sigmoid)
        probabilities = torch.sigmoid(logits.squeeze())
        predicted_labels = (probabilities > 0.5).float()
        accuracy = (predicted_labels == labels.squeeze()).float().mean()

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        team1_features, team2_features, labels = batch

        # Forward pass (returns logits)
        logits = self(team1_features, team2_features)

        # Compute loss
        loss = self.criterion(logits.squeeze(), labels.squeeze().float())

        # Compute accuracy (convert logits to probabilities with sigmoid)
        probabilities = torch.sigmoid(logits.squeeze())
        predicted_labels = (probabilities > 0.5).float()
        accuracy = (predicted_labels == labels.squeeze()).float().mean()

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step - returns probabilities"""
        team1_features, team2_features = batch
        logits = self(team1_features, team2_features)
        # Convert logits to probabilities for predictions
        probabilities = torch.sigmoid(logits)
        return probabilities

    def configure_optimizers(self):
        """Configure optimizer with ReduceLROnPlateau scheduler"""
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # ReduceLROnPlateau: reduce LR when validation loss plateaus
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1},
        }


if __name__ == "__main__":
    # Create transformer model
    model = NBAGamePredictionTransformerModel(
        player_input_dim=20,
        player_embedding_dim=32,
        num_players=12,
        team_embedding_dim=64,
        num_heads=4,
        num_transformer_layers=2,
        dropout=0.3,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    # Test forward pass
    batch_size = 16
    team1_features = torch.randn(batch_size, 240)  # 12 players * 20 features
    team2_features = torch.randn(batch_size, 240)

    with torch.no_grad():
        win_logits = model(team1_features, team2_features)
        win_probabilities = torch.sigmoid(win_logits)

    print(f"Model output shape: {win_logits.shape}")  # Should be (16, 1)
    print(f"Example win probabilities: {win_probabilities[:5].squeeze().tolist()}")

    # Print model architecture
    print(f"\nTotal model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print component breakdown
    print(f"\nParameter breakdown:")
    print(f"  PlayerEncoder: {sum(p.numel() for p in model.team_encoder.player_encoder.parameters()):,}")
    print(f"  Transformer Encoder: {sum(p.numel() for p in model.team_encoder.transformer_encoder.parameters()):,}")
    print(f"  Team Projection: {sum(p.numel() for p in model.team_encoder.team_projection.parameters()):,}")
    print(f"  GamePredictor: {sum(p.numel() for p in model.game_predictor.parameters()):,}")
