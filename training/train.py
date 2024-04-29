import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import logging

from training.preprocess import create_pairs, create_synfing_pair, create_dataset
from training.performance_metrics import PerformanceMetrics
from training.utils import step_decay
from training.settings import IMAGE_SIZE_D, BATCH_SIZE, CLASS_SIZE, VAL_SPLIT, TB_LOG_ROOT, KERAS_MODEL_ROOT, LR_DROP, LR_EPOCHS_DROP, LR_INITIAL
from models.resnet import get_model
from deps.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

RUN_NAME = 'gaborized_resnet_giga_lr0002fr'

logger.debug(f'IMAGE_SIZE_D:{IMAGE_SIZE_D}, BATCH_SIZE:{BATCH_SIZE}, CLASS_SIZE:{CLASS_SIZE}, VAL_SPLIT:{VAL_SPLIT}')
logger.debug(f'LR_INITIAL:{LR_INITIAL}, LR_DROP:{LR_DROP}, LR_EPOCHS_DROP:{LR_EPOCHS_DROP}')

TRAIN_CLASS_SIZE = int(CLASS_SIZE*(1-VAL_SPLIT))
VAL_CLASS_SIZE = int(CLASS_SIZE*(VAL_SPLIT))
s_p_e = int(TRAIN_CLASS_SIZE//BATCH_SIZE)
v_s = int(VAL_CLASS_SIZE//BATCH_SIZE)
logger.debug(f'SPE: {s_p_e}, VS: {v_s}')

# img_paths1, img_paths2, labels = create_pairs()
# train_img1, val_img1, train_img2, val_img2, train_labels, val_labels = train_test_split(img_paths1, img_paths2, labels, test_size=VAL_SPLIT, random_state=42)

train_img1, train_img2, train_labels = create_synfing_pair(fp1='datasets\\sd04-giga\\ts\\img1', fp2='datasets\\sd04-giga\\ts\\img2', pair_len=TRAIN_CLASS_SIZE)
val_img1, val_img2, val_labels = create_synfing_pair(fp1='datasets\\sd04-giga\\vs\\img1', fp2='datasets\\sd04-giga\\vs\\img2', pair_len=VAL_CLASS_SIZE)

train_ds = create_dataset(train_img1, train_img2, train_labels, TRAIN_CLASS_SIZE, BATCH_SIZE)
val_ds = create_dataset(val_img1, val_img2, val_labels, VAL_CLASS_SIZE, BATCH_SIZE)


model = get_model(IMAGE_SIZE_D)

optomizer = Adam(learning_rate=LR_INITIAL)
model.compile(optimizer=optomizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=200, verbose=1)
checkpoint = ModelCheckpoint(os.path.join(KERAS_MODEL_ROOT, f'{RUN_NAME}.keras'), monitor='loss', save_best_only=True, verbose=1)

performance_metrics = PerformanceMetrics(val_ds, v_s)
lrate = LearningRateScheduler(step_decay)


tensorboard_callback = TensorBoard(log_dir=os.path.join(TB_LOG_ROOT, RUN_NAME), histogram_freq=1)

model.fit(
    x=train_ds, 
    epochs=100, 
    steps_per_epoch=s_p_e,
    validation_steps=v_s, 
    validation_data=val_ds, 
    verbose='auto', 
    callbacks=[checkpoint, early_stopping, lrate, performance_metrics, tensorboard_callback]
)