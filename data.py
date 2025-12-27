import os

import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset

import lightning as L

from config import Config
from utils.timefeatures import time_features


class TimeSeriesDataset(Dataset):
    """
    Sliding-window time series dataset for supervised forecasting.

    Builds input/label windows from a CSV file that includes a timestamp column,
    a target column, and optional exogenous feature columns. Optionally computes
    temporal features (e.g., hour-of-day) via `time_features`.

    Args:
        config (Config)
    """

    def __init__(self, config: Config):
        self.config = config

        if not os.path.exists(config.data.data_path):
            raise FileNotFoundError(f"{config.data.data_path} does not exist")

        data = pd.read_csv(config.data.data_path)
        columns = [
            column
            for column in list(data.columns)
            if column not in [config.data.target_column, config.data.timestamp_column]
        ]

        data = data[[config.data.timestamp_column] + columns + [config.data.target_column]]

        timestamp_features = (
            time_features(
                pd.to_datetime(data[config.data.timestamp_column].values),
                freq=config.data.freq,
            ).transpose(1, 0)
            if config.data.time_features
            else None
        )

        self.data = data.drop(columns=[config.data.timestamp_column]).values
        self.timestamp_features = timestamp_features

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve a pair of input/label windows (and optional time features).

        The input window spans `[index, index + seq_len)`. The label window spans
        `[index + seq_len - label_len, index + seq_len + pred_len)`, combining
        overlap and forecast horizon.

        Args:
            index (int): Starting index for the input window.

        Returns:
            tuple: If no timestamp features, returns
                `(x_window, y_window)` where shapes are
                `(seq_len, n_features)` and `(label_len + pred_len, n_features)`.
                If timestamp features are enabled, returns
                `(x_window, y_window, x_time, y_time)` with matching lengths.
        """
        x = index
        x_ = index + self.config.seq_len
        y = x_ - self.config.label_len
        y_ = y + self.config.label_len + self.config.pred_len

        if self.timestamp_features is None:
            return self.data[x:x_], self.data[y:y_, -1]

        return (
            self.data[x:x_],
            self.data[y:y_, -1],
            self.timestamp_features[x:x_],
            self.timestamp_features[y:y_],
        )

    def __len__(self) -> int:
        """
        Number of available sliding windows.

        Returns:
            int: Count of windows = `len(data) - seq_len - pred_len + 1`.
        """
        return len(self.data) - self.config.seq_len - self.config.pred_len + 1


class TSDataModule(L.LightningDataModule):
    """
    PyTorch Lightning data module wrapping the time series dataset.

    Splits the dataset into train/val/test subsets according to configured
    ratios and provides DataLoaders for each stage.

    Args:
        config (Config)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if config.data.train_ratio + config.data.test_ratio > 1:
            raise ValueError("train_ratio + test_ratio must not be more than 1")

        self.dataset = TimeSeriesDataset(config)

    def setup(self, stage: str | None = None) -> None:
        """
        Split dataset into train/val/test subsets.

        Args:
            stage (str | None): Lightning stage: "fit", "test", or None (both).
        """
        n = len(self.dataset)
        train = int(n * self.config.data.train_ratio)
        test = int(n * self.config.data.test_ratio)
        val = n - train - test

        if stage == "fit" or stage is None:
            self.train_dataset = Subset(self.dataset, range(train))
            self.val_dataset = Subset(self.dataset, range(train, train + val))

        if stage == "test" or stage is None:
            self.test_dataset = Subset(self.dataset, range(train + val, n))

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with shuffling."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader without shuffling."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader without shuffling."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
        )