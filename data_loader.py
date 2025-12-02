import pandas as pd

def data_loader():
    """Load or build train/val/test splits for model training/evaluation."""
    
    # 1. Load game-modeling table
    df = pd.read_parquet("datasets/games_model_df.parquet")
    print("Loaded:", df.shape)

    # 2. Robust datetime parsing (handles timezone offsets like -04:00)
    df["gameDateTimeEst"] = pd.to_datetime(
        df["gameDateTimeEst"],
        errors="coerce",      # any weird ones become NaT
        utc=True,             # keep as timezone-aware; fine for sorting
    )

    # Quick sanity check
    print("Number of NaT in gameDateTimeEst:", df["gameDateTimeEst"].isna().sum())

    # 3. Sort chronologically
    df = df.sort_values("gameDateTimeEst").reset_index(drop=True)

    print("\nSeasons present:")
    print(df["season"].value_counts().sort_index())

    # 4. Select features (home_Season_* and away_Season_*)
    feature_cols = [c for c in df.columns
                    if c.startswith("home_Season_") or c.startswith("away_Season_")]

    print("\nNumber of features:", len(feature_cols))
    print("Example features:", feature_cols[:5])

    # 5. Split by season (time-based)
    train_df = df[df["season_start_year"] <= 2016]
    val_df   = df[(df["season_start_year"] >= 2017) & (df["season_start_year"] <= 2019)]
    test_df  = df[df["season_start_year"] >= 2020]

    print(f"\nTrain: {train_df.shape}")
    print(f"Val:   {val_df.shape}")
    print(f"Test:  {test_df.shape}")

    # 6. Extract X/y for each split
    X_train = train_df[feature_cols]
    y_train = train_df["home_win"].astype(int)

    X_val = val_df[feature_cols]
    y_val = val_df["home_win"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["home_win"].astype(int)

    X_train.to_parquet("datasets/X_train.parquet")
    y_train.to_frame(name="home_win").to_parquet("datasets/y_train.parquet")

    X_val.to_parquet("datasets/X_val.parquet")
    y_val.to_frame(name="home_win").to_parquet("datasets/y_val.parquet")

    X_test.to_parquet("datasets/X_test.parquet")
    y_test.to_frame(name="home_win").to_parquet("datasets/y_test.parquet")

    print("\nSaved train/val/test splits.")

    X_train = pd.read_parquet("datasets/X_train.parquet")
    y_train = pd.read_parquet("datasets/y_train.parquet")["home_win"]

    X_val = pd.read_parquet("datasets/X_val.parquet")
    y_val = pd.read_parquet("datasets/y_val.parquet")["home_win"]

    X_test = pd.read_parquet("datasets/X_test.parquet")
    y_test = pd.read_parquet("datasets/y_test.parquet")["home_win"]
    
    return X_train, y_train, X_val, y_val, X_test, y_test