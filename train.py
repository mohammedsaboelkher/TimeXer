import hydra
from hydra.utils import instantiate, call

from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    module, scaler = call(cfg.data)
    experiment = instantiate(cfg.experiment, scaler=scaler)
    callbacks = [instantiate(cb) for cb in cfg.callbacks.values() if cb is not None]
    trainer = instantiate(cfg.trainer, callbacks=callbacks)

    print(callbacks)

    trainer.fit(
        experiment,
        datamodule=module
    )

    trainer.test(
        experiment,
        datamodule=module,
        ckpt_path="best"
    )

if __name__ == "__main__":
    main()