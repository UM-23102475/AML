class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): Number of epochs to wait after the last improvement.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def check(self, val_loss):
        """
        Check if training should stop based on validation loss.
        Args:
            val_loss (float): Current epoch's validation loss.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True