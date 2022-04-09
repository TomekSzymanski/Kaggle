import subprocess
import sys

subprocess.call(['pip', 'install', 'tensorflow_addons'])
subprocess.call(['pip', 'install', 'keras'])
subprocess.call(['pip', 'install', 'cloudml-hypertune'])
subprocess.call(['pip', 'install', '-U', 'tensorboard_plugin_profile'])

import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras import backend as K
import argparse
import hypertune
from tensorflow.keras import mixed_precision

import os
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

mixed_precision.set_global_policy('mixed_float16')

GCS_BUCKET = 'gs://deep-learning-experiments/'
GCS_PATH_FOR_DATA = GCS_BUCKET + 'mnist_preprocessed_input_data/'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--use_checkpoints', default=False, action="store_true")
parser.add_argument('--base_dropout', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=0.0001)

args = parser.parse_args()

def extract(example):
  data = tf.io.parse_example(
    example,
    {
      'image': tf.io.FixedLenFeature(shape=(32, 32, 3), dtype=tf.float32),
      'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
    }
  )
  return data['image'], data['label']

def get_dataset(filename):
  return tf.data.TFRecordDataset([GCS_PATH_FOR_DATA + filename]) \
    .map(extract, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(1024) \
    .batch(256) \
    .cache() \
    .prefetch(tf.data.experimental.AUTOTUNE)

train_dataset = get_dataset('train.tfrecord')
val_dataset = get_dataset('val.tfrecord')
test_dataset = get_dataset('test.tfrecord')


from tensorflow.keras import layers, models, losses
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers

train_shape = (32, 32, 3)
NUM_CLASSES = 10

def create_model():
    m = Sequential()

    m.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", input_shape = train_shape, kernel_regularizer=regularizers.l2(args.weight_decay)))
    m.add(LeakyReLU(0.1))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=32, kernel_size=(3,3), padding="same", input_shape = train_shape, kernel_regularizer=regularizers.l2(args.weight_decay)))
    m.add(LeakyReLU(0.1))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    m.add(Dropout(args.base_dropout))

    m.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", input_shape = train_shape, kernel_regularizer=regularizers.l2(args.weight_decay)))
    m.add(LeakyReLU(0.1))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", input_shape = train_shape, kernel_regularizer=regularizers.l2(args.weight_decay)))
    m.add(LeakyReLU(0.1))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    m.add(Dropout(args.base_dropout + 0.1))

    m.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", input_shape = train_shape, kernel_regularizer=regularizers.l2(args.weight_decay)))
    m.add(LeakyReLU(0.1))
    m.add(BatchNormalization())
    m.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", input_shape = train_shape, kernel_regularizer=regularizers.l2(args.weight_decay)))
    m.add(LeakyReLU(0.1))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    m.add(Dropout(args.base_dropout + 0.2))

    m.add(Flatten())

    m.add(Dense(256))
    m.add(LeakyReLU(0.1))
    m.add(Dropout(0.5))
    m.add(Dense(NUM_CLASSES))
    m.add(Activation("softmax", dtype='float32'))

    m.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return m

GCS_PATH_FOR_CHECKPOINTS = GCS_BUCKET + 'training_checkpoints/'
GCS_PATH_FOR_SAVED_MODEL = GCS_BUCKET + 'models'
CHECKPOINTS_PREFIX = 'checkpoint_'

def lr_schedule(epoch):
        lrate = 0.001
        if epoch > 75:
            lrate = 0.0005
        if epoch > 100:
            lrate = 0.0003
        return lrate

import datetime
log_dir = os.environ['AIP_TENSORBOARD_LOG_DIR'] if 'AIP_TENSORBOARD_LOG_DIR' in os.environ else os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1, write_steps_per_second=True, profile_batch = '1570,1700')
    
mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
with mirrored_strategy.scope():
    model = create_model()
    
    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_schedule), 
        tensorboard_callback]

    if args.use_checkpoints:
        # Restore from the latest checkpoint if available.
        latest_ckpt = tf.train.latest_checkpoint(GCS_PATH_FOR_CHECKPOINTS)
        if latest_ckpt:
            model.load_weights(latest_ckpt)

        # Create a callback to store a check at the end of each epoch.
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
          filepath=GCS_PATH_FOR_CHECKPOINTS + CHECKPOINTS_PREFIX,
          monitor='val_loss',
          save_weights_only=True
        )
        callbacks.append(ckpt_callback)

    history = model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, callbacks=callbacks)

    hpt = hypertune.HyperTune()
    score = history.history['accuracy'][-1]
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='accuracy', metric_value=score)

    # Export the model to GCS.
    model.save(GCS_PATH_FOR_SAVED_MODEL)
