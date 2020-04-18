import os
import numpy as np
import tensorflow as tf
print('using tensorflow version: ', tf.__version__)
print('tf.keras version: ', tf.keras.__version__)
from tensorflow.python.lib.io import file_io
import argparse
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# tf.keras imports
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Activation, Dropout, Flatten, BatchNormalization

MODEL_NAME = 'sample_cnn'
MODEL_NAME_H5 = MODEL_NAME + '.h5'


def auc(Y, y):
  auc = tf.metrics.auc(Y, y)[1]
  K.get_session().run(tf.local_variables_initializer())
  return auc

def build_model(input_shape):
    # Model Definition
    activ = 'relu'
    init = 'he_uniform'
    num_tags = 4

    input = tf.keras.Input(shape=(input_shape))

    conv0 = Conv1D(filters=128, kernel_size=3, strides=3, padding='valid', kernel_initializer=init)(input)
    bn0 = BatchNormalization()(conv0)
    activ0 = Activation(activ)(bn0)

    conv1 = Conv1D(128, 3, padding='same', kernel_initializer=init)(activ0)
    bn1 = BatchNormalization()(conv1)
    activ1 = Activation(activ)(bn1)
    MP1 = MaxPooling1D(pool_size=3)(activ1)

    conv2 = Conv1D(128, 3, padding='same', kernel_initializer=init)(MP1)
    bn2 = BatchNormalization()(conv2)
    activ2 = Activation(activ)(bn2)
    MP2 = MaxPooling1D(pool_size=3)(activ2)

    conv3 = Conv1D(256, 3, padding='same', kernel_initializer=init)(MP2)
    bn3 = BatchNormalization()(conv3)
    activ3 = Activation(activ)(bn3)
    MP3 = MaxPooling1D(pool_size=3)(activ3)

    conv4 = Conv1D(256, 3, padding='same', kernel_initializer=init)(MP3)
    bn4 = BatchNormalization()(conv4)
    activ4 = Activation(activ)(bn4)
    MP4 = MaxPooling1D(pool_size=3)(activ4)

    conv5 = Conv1D(256, 3, padding='same', kernel_initializer=init)(MP4)
    bn5 = BatchNormalization()(conv5)
    activ5 = Activation(activ)(bn5)
    MP5 = MaxPooling1D(pool_size=3)(activ5)

    conv6 = Conv1D(256, 3, padding='same', kernel_initializer=init)(MP5)
    bn6 = BatchNormalization()(conv6)
    activ6 = Activation(activ)(bn6)
    MP6 = MaxPooling1D(pool_size=3)(activ6)

    conv7 = Conv1D(256, 3, padding='same', kernel_initializer=init)(MP6)
    bn7 = BatchNormalization()(conv7)
    activ7 = Activation(activ)(bn7)
    MP7 = MaxPooling1D(pool_size=3)(activ7)

    conv8 = Conv1D(256, 3, padding='same', kernel_initializer=init)(MP7)
    bn8 = BatchNormalization()(conv8)
    activ8 = Activation(activ)(bn8)
    MP8 = MaxPooling1D(pool_size=3)(activ8)

    conv9 = Conv1D(512, 3, padding='same', kernel_initializer=init)(MP8)
    bn9 = BatchNormalization()(conv9)
    activ9 = Activation(activ)(bn9)
    MP9 = MaxPooling1D(pool_size=3)(activ9) # use for 59050 samples

    conv10 = Conv1D(512, 1, padding='same', kernel_initializer=init)(MP9)
    bn10 = BatchNormalization()(conv10)
    activ10 = Activation(activ)(bn10)
    dropout1 = Dropout(0.5)(activ10)

    flattened = Flatten()(dropout1)

    output = Dense(num_tags, activation='sigmoid')(flattened)

    model = tf.keras.Model(input, output)

    # Print model metadata
    print('input shape:', input_shape)
    for i, layer in enumerate(model.layers):
      print('layer {} shape: {}'.format(i, layer.get_output_at(0).get_shape().as_list()))
    model.summary()

    return model


class CloudModelCheckpoint(tf.keras.callbacks.Callback):
  """Save the model after every epoch.
  `filepath` can contain named formatting options,
  which will be filled the value of `epoch` and
  keys in `logs` (passed in `on_epoch_end`).
  For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
  then the model checkpoints will be saved with the epoch number and
  the validation loss in the filename.
  Arguments:
      filepath: string, path to save the model file.
      monitor: quantity to monitor.
      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`,
          the latest best model according to
          the quantity monitored will not be overwritten.
      mode: one of {auto, min, max}.
          If `save_best_only=True`, the decision
          to overwrite the current save file is made
          based on either the maximization or the
          minimization of the monitored quantity. For `val_acc`,
          this should be `max`, for `val_loss` this should
          be `min`, etc. In `auto` mode, the direction is
          automatically inferred from the name of the monitored quantity.
      save_weights_only: if True, then only the model's weights will be
          saved (`model.save_weights(filepath)`), else the full model
          is saved (`model.save(filepath)`).
      period: Interval (number of epochs) between checkpoints.
  """

  def __init__(self,
               filepath,
               job_dir,
               monitor='val_loss',
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               period=1,
               ):
    super(CloudModelCheckpoint, self).__init__()
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = filepath
    self.job_dir = job_dir
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.period = period
    self.epochs_since_last_save = 0

    if mode not in ['auto', 'min', 'max']:
      logging.warning('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.Inf
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.Inf
      else:
        self.monitor_op = np.less
        self.best = np.Inf

  def copy_local_save_to_cloud(self, filepath):
    with file_io.FileIO(filepath, 'r') as infile:
       with file_io.FileIO(self.job_dir + filepath, mode='w+') as outfile:
           outfile.write(infile.read())

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    self.epochs_since_last_save += 1
    if self.epochs_since_last_save >= self.period:
      self.epochs_since_last_save = 0
      filepath = self.filepath.format(epoch=epoch + 1, **logs)
      if self.save_best_only:
        current = logs.get(self.monitor)
        if current is None:
          logging.warning('Can save best model only with %s available, '
                          'skipping.', self.monitor)
        else:
          if self.monitor_op(current, self.best):
            if self.verbose > 0:
              print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                    ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                             current, filepath))
            self.best = current
            if self.save_weights_only:
              self.model.save_weights(filepath, overwrite=True)
              self.copy_local_save_to_cloud(filepath)
            else:
              self.model.save(filepath, overwrite=True)
              self.copy_local_save_to_cloud(filepath)
          else:
            if self.verbose > 0:
              print('\nEpoch %05d: %s did not improve from %0.5f' %
                    (epoch + 1, self.monitor, self.best))
      else:
        if self.verbose > 0:
          print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
        if self.save_weights_only:
          self.model.save_weights(filepath, overwrite=True)
          self.copy_local_save_to_cloud(filepath)
        else:
          self.model.save(filepath, overwrite=True)
          self.copy_local_save_to_cloud(filepath)


def train(model, job_dir, X_train, y_train, X_test, y_test):
    lr = 0.01
    lrdecay = 1e-6
    sgd = tf.keras.optimizers.SGD(lr=lr, decay=lrdecay,momentum=0.9, nesterov=True)
    model.compile(
        loss='binary_crossentropy',
        optimizer=sgd,
        metrics=['categorical_accuracy', auc])

    checkpoint_path = MODEL_NAME + '-{epoch:02d}-{val_loss:.2f}.h5'
    checkpointer = CloudModelCheckpoint(filepath=checkpoint_path, job_dir=job_dir, verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    callbacks_list = [checkpointer]

    model.fit(x=X_train, y=y_train,
              batch_size=23,
              epochs=32,
              validation_data=(X_test, y_test),
              callbacks=callbacks_list,
              verbose=2
              )

    tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference

    # cloud save to .h5
    print('saving to .h5 job_dir bucket')
    model.save(MODEL_NAME_H5) # local save
    with file_io.FileIO(MODEL_NAME_H5, 'r') as infile:
       with file_io.FileIO(job_dir + MODEL_NAME_H5, mode='w+') as outfile:
           outfile.write(infile.read())

    # cloud export to SavedModel
    export_path = job_dir + MODEL_NAME + '/1'
    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    print('saving SavedModel to job_dir bucket at:', export_path)
    # tf.contrib.saved_model.save_keras_model(model, export_path) # experimental in tensorflow>1.12
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input': model.input},
            outputs={'output': model.output})


def main(job_dir, **args):
    # # local load
    X = np.load(os.getcwd() + '/x_mood_dataset_samples_22k_off43_dur59049.npy')
    y = np.load(os.getcwd() + '/y_mood_dataset_samples_22k_off43_dur59049.npy')

    # cloud load (PYTHON 2 ONLY - py2.7 runs on CloudML by default)
    # from StringIO import StringIO
    # f = StringIO(file_io.read_file_to_string(job_dir + 'x_mood_dataset_samples_22k_off43_dur59049.npy'))
    # X = np.load(f)
    # f1 = StringIO(file_io.read_file_to_string(job_dir + 'y_mood_dataset_samples_22k_off43_dur59049.npy'))
    # y = np.load(f1)

    print('shape of training data: ', X.shape)
    print('shape of labels: ', y.shape)

    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    model = build_model(input_shape=X_train[0].shape)
    # train(model, job_dir, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', help='GCS location where you store export checkpoints & store models', required=True)
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
