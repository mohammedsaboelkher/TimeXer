import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    """
    Instance normalization module for normalizing and denormalizing tensors.

    This module computes per-instance and per-series statistics (mean and standard deviation)
    can be used to normalize or denormalize data.

    Args:
        eps (float): Small epsilon value added for numerical stability when
            computing the standard deviation. Default: 1e-5.
    """
    def __init__(
            self,
            eps: float = 1e-5,
        ):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, norm: bool = True) -> torch.Tensor:
        """
        Normalize or denormalize the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape. When `norm=True`,
                statistics are computed along dimension 1.
            norm (bool): If True, compute normalization statistics and normalize
                the tensor. If False, denormalize using previously stored
                statistics. Default: True.

        Returns:
            torch.Tensor: Normalized or denormalized tensor with the same shape as input.
        """
        if norm:
            self._get_statistics(x)
            x = self._normalize(x)
        else:
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor) -> None:
        """
        Compute and store mean and standard deviation statistics.

        Computes the mean and standard deviation of the input tensor along
        the sequence length, with keepdim=True to preserve dimensions.
        Results are detached from the computation graph.
    
        Args:
            x (torch.Tensor): Input tensor from which to compute statistics.
        """
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor using stored mean and standard deviation.

        Performs standard normalization: (x - mean) / std using the statistics
        computed by `_get_statistics`.

        Args:
            x (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: Normalized tensor with zero mean and unit variance
                along the normalized dimension.
        """
        x = x - self.mean
        x = x / self.std
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse the normalization using stored mean and standard deviation.

        Performs the inverse operation of normalization: x * std + mean,
        restoring the original scale and shift of the data.

        Args:
            x (torch.Tensor): Normalized tensor to denormalize.

        Returns:
            torch.Tensor: Denormalized tensor in the original scale.
        """
        x = x * self.std
        x = x + self.mean
        return x