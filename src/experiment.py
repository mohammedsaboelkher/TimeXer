from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import lightning as L

from sklearn.preprocessing import StandardScaler


class Experiment(L.LightningModule):
    """
    TimeXer training experiment

    Args:
        model (TimeXer): The TimeXer model to be trained.
        optimizer (Optimizer): Optimizer for training the model.
        scheduler (Optional[LRScheduler]): Learning rate scheduler, or None.
        scaler (StandardScaler): Fitted scaler for denormalizing predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        scaler: StandardScaler
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

    def forward(
        self,
        endo: torch.Tensor,
        exo: torch.Tensor,
        ext: torch.Tensor | None = None
    ) -> torch.Tensor:
        """ 
        Args:
            endo (torch.Tensor): Endogenous (target) input tensor of shape
                `(batch, seq_len)` representing the univariate time series to be forecasted.
            exo (torch.Tensor): Exogenous (covariate) input tensor of shape
                `(batch, seq_len, n_vars)` containing auxiliary variables that
                may help improve predictions.
            ext (torch.Tensor | None): Optional temporal markers or features of shape
                `(batch, seq_len, n_ext_vars)` that provide additional context.

        Returns:
            torch.Tensor: Forecasted values of shape
                `(batch, pred_len)` where `pred_len` is overlap + horizon.
        """

        return self.model(endo, exo, ext)

    def _split_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Split a batch into endogenous, exogenous, external, and target components.
        
        Args:
            batch: A batch from the DataLoader containing (x, y, x_time_features, y_time_features).

        Returns:
            tuple: (endo, exo, ext, y) where endo is the target feature,
                exo are covariate features, ext are external features, and y is the target.
        """
        x, y, ext, _ = batch

        endo = x[..., -1]
        exo = x[..., :-1]

        return endo, exo, ext, y

    def _step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute model predictions and loss for a batch.
        
        Args:
            batch: A batch from the DataLoader.

        Returns:
            tuple: (loss, predictions, targets)
        """
        endo, exo, ext, y = self._split_batch(batch)
        preds = self.model(endo, exo, ext)

        return F.mse_loss(preds, y), preds, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._step(batch)

        preds = preds.detach().cpu().numpy() * self.scaler.scale_[-1] + self.scaler.mean_[-1]
        y = y.detach().cpu().numpy() * self.scaler.scale_[-1] + self.scaler.mean_[-1]

        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        self.log(
            "unnormalized_test_loss",
            F.mse_loss(torch.tensor(preds), torch.tensor(y)),
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            dict or Optimizer: Optimizer configuration, optionally with scheduler.
        """
        optimizer = self.optimizer(params=self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }

        return optimizer
