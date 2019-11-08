"""
Implementation of KAIST's best performing sample-level deep cnn
From paper "SampleCNN: End to End Deep CNN music classification"
Author: Luis Gomez
Email: luis.moodindustries@gmail.com

Model naming scheme:
SampleCNN_KAIST_31104_12khz_off57-NORM-testdata-binary-sigmoid-AFTER50.h5
^ the model     ^samps ^sr^ ^offset    ^mnual test dataset, others were trained with keras validation_split=0.1
                                             ^ cost func ^ output activation
"""
import pandas as pd
import augmentors
import numpy as np
import os
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

FEATURE_FILE_PATH = os.path.join(os.path.abspath(os.pardir), 'local_data')
TRAIN_DATA = os.path.join(FEATURE_FILE_PATH, 'train_samples_31104_12khz_off57-NORM.msg')
TEST_DATA = os.path.join(FEATURE_FILE_PATH, 'test_samples_31104_12khz_off57-NORM.msg')
MODEL_SAVE_PATH = os.path.join(FEATURE_FILE_PATH, 'SampleCNN_KAIST_31104_12khz_off57-NORM-testdata-binary-sigmoid-checkpoint-data-aug.h5')


def load_msgpack(fn):
    # we want to get samples into a specifically sized np array
    features = np.empty((0, 31104))
    labels = np.empty(0, dtype=np.int)
    feature_file_path = os.path.join(FEATURE_FILE_PATH, fn)
    msgpack_data = pd.read_msgpack(feature_file_path)
    for counter, song in enumerate(msgpack_data):
        samples = song['samples']
        #print('Song ' + song['song'] + ' has ' + str(samples.size) + ' samples')
        # put samples into hstack to give dimensions (0, x), allows you to vstack songs on top of each other and feed into net 1 by 1
        # TODO find a faster data structure to use than hstack -> vstack
        parsed_features = np.hstack(samples)
        features = np.vstack([features, parsed_features])
        # recall that 'mood' is an attribute in each msgpack song object; looks like 'angry-0'
        mood_label = song['mood'].split('-')[1]
        labels = np.append(labels, mood_label)

        # DEBUG: limit train and test set to certain amount of songs
        # if counter == 4:
        #     break

    return np.array(features), np.array(labels, dtype=np.int)


def train_generator_from_samples(x_train, y_train, batch_size):
    """Indefinitely returns a training batch when passed training samples and labels
    Args:
        x_train(ndarray): samples from a portion of a song with shape (num_songs, [samples], 1)
        y_train(ndarray): labels for each group of samples in x_train with shape (num_songs, 4)
        batch_size(int): Number of batches in an epoch
    """
    # Make an empty array to hold a batch of features and labels
    batch_features = np.zeros((batch_size, x_train.shape[1], 1))
    batch_labels = np.zeros((batch_size, y_train.shape[1]))

    # apply data augmentation
    while True:
        # choose a random song out of x_train
        for i in range(batch_size):
            rand_index = np.random.choice(len(x_train), 1)
            rand_shift = np.random.randint(-2, 3) # pitch shift range: [-2, 2]
            batch_features[i] = augmentors.pitch_shift(x_train[rand_index[0]], sr=12000, semitones=rand_shift)
            batch_labels[i] = y_train[rand_index]
        yield batch_features, batch_labels
    # while True:
        # get a random 31104 samples from the song
        # yield batch_features, batch_labels


def train(num_classes, x_train, y_train, x_test, y_test):
    if os.path.isfile(MODEL_SAVE_PATH):
        resume = input('Do you want to resume training the model {}?\nY/N: '.format(MODEL_SAVE_PATH.split('\\')[-1:]))
        if resume.lower() == 'y':
            refit_model(MODEL_SAVE_PATH, x_train, y_train, x_test, y_test)
            exit()
        else:
            exit()

    # Hyperparameters
    num_tags = num_classes
    epochs = 50 # sampleCNN used 1000 epochs
    batch_size = 23
    lr = 0.01
    lrdecay = 1e-6
    conv_window_size = 3
    sample_length = x_train.shape[1]
    # Conv1D requires 3 dimensions
    # the first dimension will change based on batch size so we do not pass it to shape argument
    # second dimension is 59050 (# samples extracted from each song)
    # and since we need a third dimesnion we make it 1 to show that each sample has no other value associated with it/ each sample is just a value in one dimension
    activ = 'relu'
    init = 'he_uniform'

    ''' NETWORK BELOW '''
    pool_input = Input(shape=(sample_length, 1))

    conv0 = Conv1D(filters=128, kernel_size=conv_window_size, strides=3, padding='valid', kernel_initializer=init)(pool_input)
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

    conv5 = Conv1D(256, 2, padding='same', kernel_initializer=init)(MP4)
    bn5 = BatchNormalization()(conv5)
    activ5 = Activation(activ)(bn5)
    MP5 = MaxPooling1D(pool_size=2)(activ5)

    conv6 = Conv1D(256, 4, padding='same', kernel_initializer=init)(MP5)
    bn6 = BatchNormalization()(conv6)
    activ6 = Activation(activ)(bn6)
    MP6 = MaxPooling1D(pool_size=4)(activ6)

    conv7 = Conv1D(256, 4, padding='same', kernel_initializer=init)(MP6)
    bn7 = BatchNormalization()(conv7)
    activ7 = Activation(activ)(bn7)
    MP7 = MaxPooling1D(pool_size=4)(activ7)

    conv8 = Conv1D(256, 2, padding='same', kernel_initializer=init)(MP7)
    bn8 = BatchNormalization()(conv8)
    activ8 = Activation(activ)(bn8)
    MP8 = MaxPooling1D(pool_size=2)(activ8)

    conv9 = Conv1D(512, 2, padding='same', kernel_initializer=init)(MP8)
    bn9 = BatchNormalization()(conv9)
    activ9 = Activation(activ)(bn9)
    MP9 = MaxPooling1D(pool_size=2)(activ9)  # used for 12000 sample rate only (32136 samples)
    # MP9 = MaxPooling1D(pool_length=1)(activ9) # used for 12000 sample rate only (32136 samples) and everything else left at 3
    # MP9 = MaxPooling1D(pool_length=3)(activ9) # use for 59050 samples

    conv10 = Conv1D(512, 3, padding='same', kernel_initializer=init)(MP9)
    bn10 = BatchNormalization()(conv10)
    activ10 = Activation(activ)(bn10)
    dropout1 = Dropout(0.5)(activ10)

    Flattened = Flatten()(dropout1)

    output = Dense(num_tags, activation='sigmoid')(Flattened)
    model = Model(input=pool_input, output=output)

    sgd = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])

    checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    callbacks_list = [checkpointer]

    # model.fit(x=x_train, y=y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_data=(x_test, y_test),
    #           callbacks=callbacks_list,
    #           verbose=1
    #           )

    model.fit_generator(
        train_generator_from_samples(x_train, y_train, batch_size),
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        verbose=1,
        epochs=epochs,
        steps_per_epoch=x_train.shape[0]/batch_size * 3, # * 3 gives data augmentation a good chance to work on each song in each epoch
        )
    return


def refit_model(model_path, x_train, y_train, x_test, y_test):
    model = keras.models.load_model(model_path)
    epochs = 50
    batch_size = 23
    checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    callbacks_list = [checkpointer]

    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_data=(x_test, y_test),
    #           callbacks=callbacks_list,
    #           verbose=1
    #           )

    model.fit_generator(
        train_generator_from_samples(x_train, y_train, batch_size),
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        verbose=1,
        epochs=epochs,
        steps_per_epoch=x_train.shape[0]/batch_size * 3
    )
    return


if __name__ == '__main__':
    # Load Data
    num_classes = 4
    print('\nLoading training data...')
    x_train, raw_train_labels = load_msgpack(TRAIN_DATA)
    y_train = keras.utils.to_categorical(raw_train_labels, num_classes)

    print('\nLoading testing data...')
    x_test, raw_test_labels = load_msgpack(TEST_DATA)
    y_test = keras.utils.to_categorical(raw_test_labels, num_classes)

    # Reshape Data (Conv1D requires three dimensions)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    print('Dimensionality of training data: ', x_train.shape)
    print('Dimensionality of training labels: ', y_train.shape)

    # Train
    train(num_classes, x_train, y_train, x_test, y_test)
