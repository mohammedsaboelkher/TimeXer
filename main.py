import dataclasses

import tyro

import torch
import torch.nn as nn

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from architecture import TimeXer
from config import Config
from data import TSDataModule


class TimeXerModule(L.LightningModule):

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.model = TimeXer(config)
        self.loss_fn = nn.MSELoss()

        self.save_hyperparameters(dataclasses.asdict(config))

    def forward(self, endo: torch.Tensor, exo: torch.Tensor, marks: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(endo, exo, marks)

    def _split_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x, y, x_marks, y_marks = batch

        endo = x[..., -1]
        exo = x[..., :-1]

        if exo.shape[-1] == 0:
            exo = torch.zeros((*exo.shape[:-1], 1), device=exo.device, dtype=exo.dtype)

        return endo, exo, x_marks, y

    def _step(self, batch) -> torch.Tensor:
        endo, exo, x_marks, y = self._split_batch(batch)
        preds = self.model(endo, exo, x_marks)

        return self.loss_fn(preds, y)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=max(1, self.config.training.patience // 2),
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",
            },
        }


def main(config: Config) -> None:
    datamodule = TSDataModule(config)
    model = TimeXerModule(config)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.training.patience, mode="min"),
        ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="timexer-{epoch:02d}-{val_loss:.4f}"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = TensorBoardLogger(save_dir="runs", name="timexer")

    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        gradient_clip_val=config.training.gradient_clip_val,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
