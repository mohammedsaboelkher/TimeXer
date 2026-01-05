from torch.utils.data import Dataset, DataLoader

import lightning as L


class TSDataModule(L.LightningDataModule):
    """ 
    Args:
        train (Dataset): Training dataset.
        val (Dataset): Validation dataset.
        test (Dataset): Test dataset.
        batch_size (int): Batch size for all DataLoaders. Default: 32.
    """
    def __init__(
        self,
        train: Dataset, 
        val: Dataset, 
        test: Dataset,
        batch_size: int = 32,
    ):
        super().__init__()

        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )