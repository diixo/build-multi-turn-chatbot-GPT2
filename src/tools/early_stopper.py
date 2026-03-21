from utils import LOGGER



class EarlyStopper:
    def __init__(self, patience=0):
        self.best_epoch = 0
        self.is_init = True
        self.best_high, self.best_low = 0, float('inf')
        self.patience = patience


    def __call__(self, epoch, high=None, low=None):
        if self.is_init:
            self.is_init = False
            self.best_high, self.bestlow = high, low
            return False

        if self.patience == 0:
            return False
        
        # update best metrics
        is_high_updated, is_low_updated = False, False
        if (self.best_high is not None) and (high is not None) and (high > self.best_high):
            self.best_high = high
            is_high_updated = True
        
        if (self.best_low is not None) and (low is not None) and (low < self.best_low):
            self.best_low = low
            is_low_updated = True
        
        # patience update
        if any([is_high_updated, is_low_updated]):
           self.best_epoch = epoch

        delta = epoch - self.best_epoch     # epochs without improvement
        stop = delta >= self.patience    # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `patience=300` or use `patience=0` to disable EarlyStopping.')
        return stop