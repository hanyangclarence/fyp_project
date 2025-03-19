

from pytorch_lightning.callbacks import Callback

class DummyLogger(Callback):
    def __init__(self, save_video):
        super().__init__()
        self.save_video = save_video

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass