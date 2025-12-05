# NBA Game Prediction

A deep learning model that predicts NBA game outcomes using player statistics.

## Prerequisites

### Download Dataset

Download the NBA dataset from Kaggle:
[Historical NBA Data and Player Box Scores](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores/data)

**Manual Setup:**
1. Download the dataset from the link above
2. Extract the downloaded ZIP file
3. Rename the extracted folder to `nba_dataset`
4. Move it into the `data/` directory

Your directory structure should look like:
```
project/
└── data/
    └── nba_dataset/
        ├── PlayerStatistics.csv
        ├── TeamStatistics.csv
        └── ... (other CSV files)
```

## Quick Start
### 0. Set up Conda Environment (Recommended)**

```bash
conda create -n nba_pred python=3.11
conda activate nba_pred
pip install -r requirements.txt
```

### 1. Data Preparation

After downloading the dataset, preprocess the raw player statistics and generate season averages:

```bash
cd data
python preprocess_season_data.py
```

This creates `Season_player_data.csv` with player season averages (2000-2025). (Takes about 1 hour)

### 2. Feature Precomputation

Precompute features for faster training: (Takes about 1 hour for each)

```bash
# Features with player stats only
python precompute_features.py

# Features with team info
python precompute_features_with_teaminfo.py
```

#### Data Split

- **Train + Validation**: 2000-2024 seasons (90% / 10% split, randomly shuffled)
- **Test**: 2024-2025 season

### 3. Training

Return to the project root and train the model:

```bash
cd ..

# Train baseline model
python baseline_logistic.py

# Train basic model
python train.py

# Train transformer enhanced model with players info only
python train_transformer.py

# Train with team info
python train_transformer_with_teaminfo.py
```

### 4. Model and Feature Analysis

run model-feature-analysis.ipynb


## Requirements

- Python 3.8+
- See `requirements.txt` for full list
