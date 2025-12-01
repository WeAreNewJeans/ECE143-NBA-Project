import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PlayerEncoder(nn.Module):
    """
    Encodes individual player feature vectors (Lightweight version)
    Input: 20-dim (10 season avg features + 10 recent 5-game avg features)
    Output: 16-dim embedding
    Architecture: 20 -> 32 -> 16
    """
    def __init__(self, input_dim=20, hidden_dim=32, output_dim=16):
        super(PlayerEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
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


class TeamEncoder(nn.Module):
    """
    Aggregates 12 players' feature vectors into a single team vector (Lightweight version)
    Input: 240-dim (12 players * 20-dim features)
    Output: 32-dim team embedding
    Architecture: 192 (12*16) -> 64 -> 32
    """
    def __init__(self, player_input_dim=20, player_embedding_dim=16,
                 num_players=12, team_output_dim=32):
        super(TeamEncoder, self).__init__()
        self.num_players = num_players
        self.player_input_dim = player_input_dim
        self.player_embedding_dim = player_embedding_dim

        # Shared player encoder (20 -> 32 -> 16)
        self.player_encoder = PlayerEncoder(
            input_dim=player_input_dim,
            hidden_dim=32,
            output_dim=player_embedding_dim
        )

        # Team aggregation layer - aggregates 12 player embeddings (192-dim) into team vector
        # Architecture: 192 -> 64 -> 32
        self.team_aggregator = nn.Sequential(
            nn.Linear(num_players * player_embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, team_output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: shape (batch_size, 240) - concatenated features of 12 players
        Returns:
            shape (batch_size, team_output_dim) - team vector
        """
        batch_size = x.size(0)

        # Reshape 240-dim to (batch_size, 12, 20)
        x = x.view(batch_size, self.num_players, self.player_input_dim)

        # Encode each player
        # Reshape to (batch_size * 12, 20) to pass through player_encoder
        x = x.view(batch_size * self.num_players, self.player_input_dim)
        player_embeddings = self.player_encoder(x)  # (batch_size * 12, player_embedding_dim)

        # Reshape back to (batch_size, 12 * player_embedding_dim)
        player_embeddings = player_embeddings.view(batch_size, self.num_players * self.player_embedding_dim)

        # Aggregate into team vector
        team_embedding = self.team_aggregator(player_embeddings)

        return team_embedding


class GamePredictor(nn.Module):
    """
    Takes two team vectors and predicts the win probability of the first team (Lightweight version)
    Architecture: 64 (2*32) -> 32 -> 1
    """
    def __init__(self, team_embedding_dim=32):
        super(GamePredictor, self).__init__()

        # Concatenate two team vectors and make prediction
        # Note: No Sigmoid here - we use BCEWithLogitsLoss which includes sigmoid
        # Architecture: 64 -> 32 -> 1
        self.predictor = nn.Sequential(
            nn.Linear(team_embedding_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)  # Output logits (not probabilities)
        )

    def forward(self, team1_embedding, team2_embedding):
        """
        Args:
            team1_embedding: shape (batch_size, team_embedding_dim) - first team vector
            team2_embedding: shape (batch_size, team_embedding_dim) - second team vector
        Returns:
            shape (batch_size, 1) - win probability of the first team
        """
        # Concatenate two team vectors
        combined = torch.cat([team1_embedding, team2_embedding], dim=1)
        win_prob = self.predictor(combined)
        return win_prob


class NBAGamePredictionModel(pl.LightningModule):
    """
    Complete NBA game prediction model (PyTorch Lightning)
    """
    def __init__(
        self,
        player_input_dim=20,
        player_embedding_dim=32,
        num_players=12,
        team_embedding_dim=128,
        learning_rate=1e-3,
        weight_decay=1e-5
    ):
        super(NBAGamePredictionModel, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Team encoder
        self.team_encoder = TeamEncoder(
            player_input_dim=player_input_dim,
            player_embedding_dim=player_embedding_dim,
            num_players=num_players,
            team_output_dim=team_embedding_dim
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
            shape (batch_size, 1) - win probability of the first team
        """
        # Encode both teams
        team1_embedding = self.team_encoder(team1_features)
        team2_embedding = self.team_encoder(team2_features)

        # Predict win probability
        win_prob = self.game_predictor(team1_embedding, team2_embedding)

        return win_prob

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
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

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
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

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
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step - returns probabilities"""
        team1_features, team2_features = batch
        logits = self(team1_features, team2_features)
        # Convert logits to probabilities for predictions
        probabilities = torch.sigmoid(logits)
        return probabilities

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


if __name__ == "__main__":
    # Create model
    model = NBAGamePredictionModel(
        player_input_dim=20,
        player_embedding_dim=32,
        num_players=12,
        team_embedding_dim=128,
        learning_rate=1e-3
    )

    # Test forward pass
    batch_size = 16
    team1_features = torch.randn(batch_size, 240)  # 12 players * 20 features
    team2_features = torch.randn(batch_size, 240)

    with torch.no_grad():
        win_probabilities = model(team1_features, team2_features)

    print(f"Model output shape: {win_probabilities.shape}")  # Should be (16, 1)
    print(f"Example win probabilities: {win_probabilities[:5].squeeze().tolist()}")

    # Print model architecture
    print(f"\nTotal model parameters: {sum(p.numel() for p in model.parameters()):,}")
