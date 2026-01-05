import os
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.data.dataset import TimeSeriesDataset
from src.data.module import TSDataModule

from util.time import time_features


def load(
    data_path: str,
    timestamp: str,
    target: Optional[str] = None,
    seq: int = 168,
    overlap: int = 0,
    horizon: int = 24,
    freq: Optional[str] = None,
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    batch_size: int = 32,
):
    """
    Load and preprocess time series data for model training.
    
    Reads a CSV file, normalizes features, splits into train/validation/test sets,
    optionally extracts temporal features, and returns a Lightning DataModule.
    
    Args:
        data_path (str): Path to the CSV file containing the time series data.
        timestamp (str): Column name containing timestamp information.
        target (Optional[str]): Column name of the target variable. If None, uses all features.
        seq (int): Sequence length for input windows.
        overlap (int): Overlap between input and output windows.
        horizon (int): Forecast horizon length.
        freq (Optional[str]): Pandas frequency string for time feature extraction (e.g., 'H', 'D').
            If None, no external temporal features are used.
        train_ratio (float): Proportion of data for training, must be in (0, 1). Default: 0.8.
        test_ratio (float): Proportion of data for testing, must be in (0, 1). Default: 0.1.
            Validation gets the remaining ratio: 1 - train_ratio - test_ratio.
        batch_size (int): Batch size for DataLoaders. Default: 32.

    Returns:
        tuple: (TSDataModule, StandardScaler) where TSDataModule contains train/val/test
            DataLoaders and StandardScaler is fitted on training data for denormalization.

    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If train_ratio or test_ratio are not in (0, 1), or if their sum exceeds 1.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} does not exist")

    if not 0 < train_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1) and test_ratio in (0, 1)")

    if train_ratio + test_ratio > 1:
        raise ValueError("train_ratio + test_ratio must not be more than 1")

    data = pd.read_csv(data_path)
    features = data.drop(columns=[timestamp])

    # Rearrange columns to have target at the end 
    features = features[[col for col in features.columns if col != target] + [target]] if target else features

    n = len(data)
    train_end = int(n * train_ratio)
    test_start = n - int(n * test_ratio)

    scaler = StandardScaler()

    train_data, val_data, test_data = np.split(
        features,
        [train_end, test_start]
    )
 
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    train_ext, val_ext, test_ext = (None, None, None) if freq is None else np.split(
        time_features(pd.to_datetime(data[timestamp].values), freq=freq).transpose(1, 0),
        [train_end, test_start]
    ) 

    datasets = [
        TimeSeriesDataset(
            data,
            ext,
            target is not None,
            seq,
            overlap,
            horizon
        ) for data, ext in zip((train_data, val_data, test_data), (train_ext, val_ext, test_ext))
    ]

    module = TSDataModule(
        train=datasets[0],
        val=datasets[1],
        test=datasets[2],
        batch_size=batch_size,
    )
    
    return module, scaler