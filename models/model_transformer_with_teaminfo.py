import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
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
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
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

    Note: Players are sorted by playing time in precompute_features_with_teaminfo.py
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
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, seq_len, d_model)
        Returns:
            shape (batch_size, seq_len, d_model) - with positional encoding added
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class TransformerTeamEncoderWithTeamInfo(nn.Module):
    """
    Uses Transformer to aggregate player embeddings + team info into team representation.

    Input:
        - player_features: (batch, 240) - 12 players * 20 features
        - is_home: (batch, 1) - home/away indicator
        - win_rate: (batch, 1) - team win rate before this game
        - season_stats: (batch, 9) - team season stats [PPG, RPG, APG, SPG, BPG, TOV, FG%, 3P%, FT%]

    Output: (batch, 64) - team embedding

    Architecture:
    1. Encode each player: 20 -> 48 -> 32
    2. Add positional encoding to distinguish player order
    3. Multi-head self-attention to capture player interactions
    4. Feed-forward network for team representation
    5. Concatenate with team info (home/away + win rate + season stats = 11-dim)
    6. Fusion layer to combine player and team features
    """
    def __init__(self, player_input_dim=20, player_embedding_dim=32,
                 num_players=12, team_output_dim=64,
                 num_heads=4, num_layers=2, dropout=0.3):
        super(TransformerTeamEncoderWithTeamInfo, self).__init__()
        self.num_players = num_players
        self.player_input_dim = player_input_dim
        self.player_embedding_dim = player_embedding_dim

        # Player encoder: 20 -> 48 -> 32
        self.player_encoder = PlayerEncoder(
            input_dim=player_input_dim,
            hidden_dim=48,
            output_dim=player_embedding_dim
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=player_embedding_dim,
            max_len=num_players
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=player_embedding_dim,
            nhead=num_heads,
            dim_feedforward=player_embedding_dim * 2,  # 64
            dropout=dropout,
            activation='relu',
            batch_first=True  # Important: (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Team aggregation: Pool transformer output
        # We'll use mean pooling over all players
        self.team_projection = nn.Sequential(
            nn.Linear(player_embedding_dim, team_output_dim // 2),  # 32
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Team info fusion layer
        # Combines player representation (32-dim) with team info (11-dim)
        # Team info: home/away (1) + win_rate (1) + season_stats (9) = 11
        self.fusion = nn.Sequential(
            nn.Linear(team_output_dim // 2 + 11, team_output_dim),  # 32 + 11 -> 64
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, player_features, is_home, win_rate, season_stats):
        """
        Args:
            player_features: shape (batch_size, 240) - concatenated features of 12 players
            is_home: shape (batch_size, 1) - home/away indicator (1=home, 0=away)
            win_rate: shape (batch_size, 1) - team win rate before this game
            season_stats: shape (batch_size, 9) - team season stats [PPG, RPG, APG, SPG, BPG, TOV, FG%, 3P%, FT%]

        Returns:
            shape (batch_size, team_output_dim) - team vector
        """
        batch_size = player_features.size(0)

        # Reshape to (batch_size, 12, 20)
        x = player_features.view(batch_size, self.num_players, self.player_input_dim)

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
        player_representation = transformer_output.mean(dim=1)

        # Project to intermediate dimension
        # (batch_size, 32) -> (batch_size, 32)
        player_representation = self.team_projection(player_representation)

        # Concatenate with team info
        # (batch_size, 32) + (batch_size, 11) -> (batch_size, 43)
        # Team info: home/away (1) + win_rate (1) + season_stats (9) = 11
        team_info = torch.cat([is_home, win_rate, season_stats], dim=1)  # (batch, 11)
        combined = torch.cat([player_representation, team_info], dim=1)  # (batch, 43)

        # Fusion layer
        # (batch_size, 43) -> (batch_size, 64)
        team_embedding = self.fusion(combined)

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
            nn.Linear(32, 1)  # Output logits (not probabilities)
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


class NBAGamePredictionTransformerModelWithTeamInfo(pl.LightningModule):
    """
    NBA game prediction model using Transformer + Team Information
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
        weight_decay=1e-4
    ):
        super(NBAGamePredictionTransformerModelWithTeamInfo, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Transformer-based team encoder with team info
        self.team_encoder = TransformerTeamEncoderWithTeamInfo(
            player_input_dim=player_input_dim,
            player_embedding_dim=player_embedding_dim,
            num_players=num_players,
            team_output_dim=team_embedding_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout
        )

        # Game predictor
        self.game_predictor = GamePredictor(team_embedding_dim=team_embedding_dim)

        # Loss function - BCEWithLogitsLoss is safe for mixed precision training
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, team1_features, team2_features,
                team1_home, team2_home,
                team1_winrate, team2_winrate,
                team1_season_stats, team2_season_stats):
        """
        Args:
            team1_features: shape (batch_size, 240) - features of the first team
            team2_features: shape (batch_size, 240) - features of the second team
            team1_home: shape (batch_size, 1) - is team1 home?
            team2_home: shape (batch_size, 1) - is team2 home?
            team1_winrate: shape (batch_size, 1) - team1 win rate
            team2_winrate: shape (batch_size, 1) - team2 win rate
            team1_season_stats: shape (batch_size, 9) - team1 season stats
            team2_season_stats: shape (batch_size, 9) - team2 season stats
        Returns:
            shape (batch_size, 1) - win logits of the first team
        """
        # Encode both teams using transformer + team info
        team1_embedding = self.team_encoder(team1_features, team1_home, team1_winrate, team1_season_stats)
        team2_embedding = self.team_encoder(team2_features, team2_home, team2_winrate, team2_season_stats)

        # Predict win probability
        win_logits = self.game_predictor(team1_embedding, team2_embedding)

        return win_logits

    def training_step(self, batch, batch_idx):
        """Training step"""
        (team1_features, team2_features, team1_home, team2_home, team1_winrate, team2_winrate,
         team1_season_stats, team2_season_stats, labels) = batch

        # Forward pass (returns logits)
        logits = self(team1_features, team2_features,
                     team1_home, team2_home,
                     team1_winrate, team2_winrate,
                     team1_season_stats, team2_season_stats)

        # Compute loss (BCEWithLogitsLoss expects logits and labels)
        loss = self.criterion(logits.squeeze(), labels.squeeze().float())

        # Compute accuracy (convert logits to probabilities with sigmoid)
        probabilities = torch.sigmoid(logits.squeeze())
        predicted_labels = (probabilities > 0.5).float()
        accuracy = (predicted_labels == labels.squeeze()).float().mean()

        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        (team1_features, team2_features, team1_home, team2_home, team1_winrate, team2_winrate,
         team1_season_stats, team2_season_stats, labels) = batch

        # Forward pass (returns logits)
        logits = self(team1_features, team2_features,
                     team1_home, team2_home,
                     team1_winrate, team2_winrate,
                     team1_season_stats, team2_season_stats)

        # Compute loss
        loss = self.criterion(logits.squeeze(), labels.squeeze().float())

        # Compute accuracy (convert logits to probabilities with sigmoid)
        probabilities = torch.sigmoid(logits.squeeze())
        predicted_labels = (probabilities > 0.5).float()
        accuracy = (predicted_labels == labels.squeeze()).float().mean()

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        (team1_features, team2_features, team1_home, team2_home, team1_winrate, team2_winrate,
         team1_season_stats, team2_season_stats, labels) = batch

        # Forward pass (returns logits)
        logits = self(team1_features, team2_features,
                     team1_home, team2_home,
                     team1_winrate, team2_winrate,
                     team1_season_stats, team2_season_stats)

        # Compute loss
        loss = self.criterion(logits.squeeze(), labels.squeeze().float())

        # Compute accuracy (convert logits to probabilities with sigmoid)
        probabilities = torch.sigmoid(logits.squeeze())
        predicted_labels = (probabilities > 0.5).float()
        accuracy = (predicted_labels == labels.squeeze()).float().mean()

        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step - returns probabilities"""
        (team1_features, team2_features, team1_home, team2_home, team1_winrate, team2_winrate,
         team1_season_stats, team2_season_stats) = batch
        logits = self(team1_features, team2_features,
                     team1_home, team2_home,
                     team1_winrate, team2_winrate,
                     team1_season_stats, team2_season_stats)
        # Convert logits to probabilities for predictions
        probabilities = torch.sigmoid(logits)
        return probabilities

    def configure_optimizers(self):
        """Configure optimizer with fixed learning rate (no scheduler)"""
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # No learning rate scheduler - use fixed LR
        return optimizer


if __name__ == "__main__":
    # Create transformer model with team info
    model = NBAGamePredictionTransformerModelWithTeamInfo(
        player_input_dim=20,
        player_embedding_dim=32,
        num_players=12,
        team_embedding_dim=64,
        num_heads=4,
        num_transformer_layers=2,
        dropout=0.3,
        learning_rate=1e-3,
        weight_decay=1e-4
    )

    # Test forward pass
    batch_size = 16
    team1_features = torch.randn(batch_size, 240)  # 12 players * 20 features
    team2_features = torch.randn(batch_size, 240)
    team1_home = torch.randint(0, 2, (batch_size, 1)).float()  # 0 or 1
    team2_home = 1.0 - team1_home  # Opposite
    team1_winrate = torch.rand(batch_size, 1)  # Random win rates [0, 1]
    team2_winrate = torch.rand(batch_size, 1)
    team1_season_stats = torch.randn(batch_size, 9)  # 9 season stats
    team2_season_stats = torch.randn(batch_size, 9)

    with torch.no_grad():
        win_logits = model(team1_features, team2_features,
                          team1_home, team2_home,
                          team1_winrate, team2_winrate,
                          team1_season_stats, team2_season_stats)
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
    print(f"  Team Info Fusion: {sum(p.numel() for p in model.team_encoder.fusion.parameters()):,}")
    print(f"  GamePredictor: {sum(p.numel() for p in model.game_predictor.parameters()):,}")
