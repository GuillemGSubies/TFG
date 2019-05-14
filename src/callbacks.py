# @author Guillem G. Subies

from keras.callbacks import ModelCheckpoint


class ModelFullCheckpoint(ModelCheckpoint):
    """Custom ModelCheckpoint that saves ALL the model, tokenizer included"""

    def __init__(self, modelo, **kwargs):
        """The same init but with an extra param"""

        super().__init__(**kwargs)
        self.modelo = modelo

    def on_epoch_end(self, epoch, logs=None):
        """https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L697
        The only change is that we perform save on self.modelo"""

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    print(
                        f"WARNIGN! Can't save best model only with {self.monitor} available, skipping."
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                f"\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.3f} to {current:.3f},"
                                f" saving model to {filepath}"
                            )
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.modelo.save(filepath)
                    else:
                        if self.verbose > 0:
                            print(f"\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.3f}")
            else:
                if self.verbose > 0:
                    print(f"\nEpoch {epoch + 1}: saving model to {filepath}")
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.modelo.save(filepath)
