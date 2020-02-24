import os
import numpy as np
import argparse
import sklearn
from sklearn.model_selection import train_test_split
# equivalent tf.keras imports
import tensorflow as tf
print('using tensorflow version: ', tf.__version__)
print('tf.keras version: ', tf.keras.__version__)
from tensorflow.python.lib.io import file_io
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization


# MODEL_SAVE_PATH = os.path.join(os.pardir, 'models', '2D_genre_cnn.h5') # for local
MODEL_NAME = '2D_mood_cnn' # for CloudML
MODEL_NAME_H5 = MODEL_NAME + '.h5'


def build_model(input_shape, using_tfkeras=True):
    # Model Definition
    num_genres = 4

    if using_tfkeras:
        model = tf.keras.Sequential()
    else:
        model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    # V1
    model.add(Dense(num_genres, activation='softmax'))

    print('input shape: ', input_shape)
    model.summary()
    return model


def train(model, job_dir, X_train, y_train, X_test, y_test):
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

    # checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    # callbacks_list = [checkpointer]

    hist = model.fit(X_train, y_train,
              batch_size=32,
              epochs=50,
              verbose=2,
              validation_data=(X_test, y_test)
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
    logs_path = job_dir + '/logs/'

    # local load
    # X = np.load('x_mood_dataset.npy')
    # y = np.load('y_mood_dataset.npy')

    # if the above doesn't work for loading, try this:
    # python 2 only
    from StringIO import StringIO
    f = StringIO(file_io.read_file_to_string(job_dir + 'x_mood_dataset.npy'))
    # print('x(mels) file handler: ', f)
    X = np.load(f)
    f1 = StringIO(file_io.read_file_to_string(job_dir + 'y_mood_dataset.npy'))
    # print('y(labels) file handler: ', f1)
    y = np.load(f1)
    print('shape of x: ', X.shape)
    print('shape of labels: ', y.shape)


    y = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = build_model(input_shape=X_train[0].shape)
    train(model, job_dir, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', help='GCS location where you store export checkpoints & store models', required=True)
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
