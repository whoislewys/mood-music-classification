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

MODEL_NAME = 'sample_cnn_v2_standardgpu'
MODEL_NAME_H5 = MODEL_NAME + '.h5'
NUM_GPUS = 4 # complex_model_m_gpu machine on CloudML has 4 GPUs


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
    model.summary()

    return model


def train(model, job_dir, X_train, y_train, X_test, y_test):
    lr = 0.01
    lrdecay = 1e-6
    sgd = tf.keras.optimizers.SGD(lr=lr, decay=lrdecay,momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])

    # checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    # callbacks_list = [checkpointer]

    model.fit(x=X_train, y=y_train,
              batch_size=23 * NUM_GPUS, # give each gpu a batch of 23!
              epochs=1,
              validation_data=(X_test, y_test),
              # callbacks=callbacks_list,
              verbose=1
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
    # X = np.load('x_mood_dataset_samples_22k_off43_dur59049.npy')
    # y = np.load('y_mood_dataset_samples_22k_off43_dur59049.npy')

    # cloud load (PYTHON 2 ONLY - py2.7 runs on CloudML by default)
    from StringIO import StringIO
    f = StringIO(file_io.read_file_to_string(job_dir + 'x_mood_dataset_samples_22k_off43_dur59049.npy'))
    X = np.load(f)
    f1 = StringIO(file_io.read_file_to_string(job_dir + 'y_mood_dataset_samples_22k_off43_dur59049.npy'))
    y = np.load(f1)

    print('shape of training data: ', X.shape)
    print('shape of labels: ', y.shape)

    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    with tf.device('/cpu:0'):
        model = build_model(input_shape=X_train[0].shape)

    model = tf.keras.utils.multi_gpu_model(model, gpus=NUM_GPUS)

    train(parakkek, job_dir, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', help='GCS location where you store export checkpoints & store models', required=True)
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
