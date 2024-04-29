import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix

import logging
logger = logging.getLogger(__name__)

class PerformanceMetrics(Callback):
    def __init__(self, val_ds, v_s):
        super(PerformanceMetrics, self).__init__()
        self.val_ds = val_ds
        self.v_s = v_s  # Number of steps to run for validation

    def on_epoch_end(self, epoch, logs=None):
        logger.info(f'Starting metric calculation for Epoch {epoch}')
        logs = logs or {}
        val_predicts = []
        val_targets = []
        
        #ctr = 0  # Counter to track the number of processed batches
        
        # Iterate through the validation dataset and stop after v_s steps
        for imgs, labels in self.val_ds.take(self.v_s):
            preds = self.model.predict(imgs, verbose=0)
            val_predicts.extend(preds)
            val_targets.extend(labels)
            #ctr += 1  # Increment counter

        val_predicts = np.array(val_predicts).squeeze()
        val_targets = np.array(val_targets)
        
        thresholds = np.arange(0.0, 1.1, 0.1)
        results = []

        # Evaluate metrics at each threshold
        for thresh in thresholds:
            binary_predictions = (val_predicts > thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(val_targets, binary_predictions, labels=[0, 1]).ravel()

            FAR = fp / (fp + tn) if (fp + tn) > 0 else 0
            FRR = fn / (fn + tp) if (fn + tp) > 0 else 0
            ACC = (tp + tn) / (tp + fp + fn + tn)

            results.append((thresh, FAR, FRR, ACC))
            logs[f'val_FAR_{thresh:.1f}'] = FAR
            logs[f'val_FRR_{thresh:.1f}'] = FRR
            logs[f'val_ACC_{thresh:.1f}'] = ACC

            logger.info(f"Threshold: {thresh:.1f}, FAR: {FAR:.3f}, FRR: {FRR:.3f}, ACC: {ACC:.3f}")