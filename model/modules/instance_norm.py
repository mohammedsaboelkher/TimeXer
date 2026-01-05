import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    """
    Instance normalization module for normalizing and denormalizing tensors.

    Args:
        dim (int | tuple[int, ...]): Dimension(s) along which to compute
            normalization statistics.
        eps (float): Small epsilon value added for numerical stability when
            computing the standard deviation. Default: 1e-5.
    """
    def __init__(
        self,
        dim: int | tuple[int, ...] = 1,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.dim = dim
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        norm: bool = True
    ) -> torch.Tensor:
        """
        Normalize or denormalize the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape. When `norm=True`,
                statistics are computed along dimension(s) specified by `dim`.
            norm (bool): If True, compute normalization statistics and normalize
                the tensor. If False, denormalize using previously stored
                statistics. Default: True.

        Returns:
            torch.Tensor: Normalized or denormalized tensor with the same shape as input.
        """
        if norm:
            self._get_statistics(x)
            return self._normalize(x)

        return self._denormalize(x)

    def _get_statistics(self, x: torch.Tensor) -> None:
        """
        Compute and store mean and standard deviation statistics.
    
        Args:
            x (torch.Tensor): Input tensor from which to compute statistics.
        """
        self.mean = torch.mean(
            x, 
            dim=self.dim,
            keepdim=True
        ).detach()

        self.std = torch.sqrt(
            torch.var(
                x, 
                dim=self.dim,
                keepdim=True,
                unbiased=False
            ) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor using stored mean and standard deviation.

        Args:
            x (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: Normalized tensor with zero mean and unit variance
            along the normalized dimension.
        """
        x -= self.mean
        x /= self.std

        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse the normalization using stored mean and standard deviation.

        Args:
            x (torch.Tensor): Normalized tensor to denormalize.

        Returns:
            torch.Tensor: Denormalized tensor in the original scale.
        """
        x *= self.std
        x += self.mean

        return x