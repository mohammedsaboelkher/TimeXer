from typing import Optional

import numpy as np

from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series forecasting with sliding windows.

    Creates overlapping windows of sequences and targets from time series data,
    optionally including external temporal features.

    Args:
        data (np.ndarray): Time series data of shape (n_samples, n_features).
        ext (Optional[np.ndarray]): External features of shape (n_samples, n_ext_features).
            If None, external features are not used. Default: None.
        target (bool): If True, returns only the last column as target; if False,
            returns all columns. Default: True.
        seq (int): Length of input sequence window. Default: 168.
        overlap (int): Overlap between input and output windows. Default: 0.
        horizon (int): Length of forecast horizon (output window length). Default: 24.
    """

    def __init__(
        self,
        data: np.ndarray,
        ext: Optional[np.ndarray] = None,
        target: bool = True,
        seq: int = 168,
        overlap: int = 0,
        horizon: int = 24,
    ):
        super().__init__()

        self.data = data.astype("float32")
        self.ext = ext.astype("float32") if ext is not None else None
        self.target = target
        self.seq = seq
        self.overlap = overlap
        self.horizon = horizon

    def __getitem__(self, index: int) -> tuple:
        """Get a sample at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: If ext is None: (input, target, None, None)
                If ext is provided: (input, target, ext_input, ext_target)
                where:
                - input: Sequence window of shape (seq, n_features)
                - target: Target window of shape (overlap + horizon,) or (overlap + horizon, n_features)
                - ext_input: External features for input window, shape (seq, n_ext_features)
                - ext_target: External features for target window, shape (overlap + horizon, n_ext_features)
        """
        x = index
        x_ = index + self.seq
        y = x_ - self.overlap
        y_ = y + self.overlap + self.horizon

        return (
            self.data[x:x_],
            self.data[y:y_, -1] if self.target else self.data[y:y_, :],
            self.ext[x:x_] if self.ext is not None else None,
            self.ext[y:y_] if self.ext is not None else None,
        )

    def __len__(self) -> int:
        return len(self.data) - self.seq - self.horizon + 1