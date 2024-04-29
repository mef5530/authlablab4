import numpy as np
from training.settings import LR_INITIAL, LR_DROP, LR_EPOCHS_DROP

def step_decay(epoch):
    lrate = LR_INITIAL * (LR_DROP ** np.floor((1+epoch)/LR_EPOCHS_DROP))
    return lrate