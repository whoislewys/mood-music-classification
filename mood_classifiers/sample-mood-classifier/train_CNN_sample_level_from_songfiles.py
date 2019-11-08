"""
this method of training will train on samples loaded directly from song files (e.g. .mp3)
"""
import augmentors
import sys
import glob
import pathlib
import numpy as np
from extract_samples import extract_random_samples
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

MOOD_DIR = pathlib.Path('..').resolve()
TRAIN_SONGS_DIR = MOOD_DIR/'songs/training_data'
TEST_SONGS_DIR = MOOD_DIR/'songs/testing_data'
try:
	Path.mkdir(MOOD_DIR/'local_data')
except:
	print('You already have a local_data folder! Setting that to the dir for saved models...')
MODEL_SAVE_PATH = MOOD_DIR/'local_data'/'train_from_songs_test.h5'

def build_song_list(parent_dir, file_ext='*.mp3'):
    # builds list of songs with assc mood labels from a parent directory
    # access each class/tag folder
    sub_dirs = [dir for dir in parent_dir.iterdir()]
    x_train = []
    y_train = []
    for sub_dir in sub_dirs:
        for song_path in sub_dir.glob(file_ext):
            song_path = str(song_path)
            x_train.append(song_path)
            # get mood label of song
            if sys.platform == 'win32':
                mood = song_path.split('\\')[7].split('-')[1]
            else:
                mood = song_path.split('/')[8].split('-')[1]
            y_train.append(mood)
    return x_train, y_train


# TODO: modify generator to use extract_random_samples on song path
def train_generator_from_song_list(x_train, y_train, batch_size):
    """Indefinitely returns a training batch when passed training samples and labels
    Args:
        x_train(ndarray): samples from a portion of a song with shape (num_songs, [samples], 1)
        y_train(ndarray): labels for each group of samples in x_train with shape (num_songs, 4)
        batch_size(int): Number of batches in an epoch
    """
    # Make an empty array to hold a batch of features and labels
    batch_features = np.zeros((batch_size, 31104, 1))
    batch_labels = np.zeros((batch_size, 4))
    while True:
        for i in range(batch_size):
            rand_index = np.random.choice(len(x_train), 1)
            samples = extract_random_samples(x_train[rand_index[0]])
            samples.shape = (samples.shape[0], 1)
            batch_features[i] = samples # can apply data augmentation on samples here
            batch_labels[i] = keras.utils.to_categorical(y_train[rand_index[0]], num_classes=4)
        yield batch_features, batch_labels


# TODO: copy modified generator & remove data augmentation to generate validation data
def test_generator_from_song_list(x_test, y_test, batch_size):
    batch_features = np.zeros((batch_size, 31104, 1))
    batch_labels = np.zeros((batch_size, 4))
    while True:
        for i in range(batch_size):
            rand_index = np.random.choice(len(x_test), 1)
            samples = extract_random_samples(x_test[rand_index[0]])
            samples.shape = (samples.shape[0], 1)
            batch_features[i] = samples  # can apply data augmentation on samples here
            batch_labels[i] = keras.utils.to_categorical(y_train[rand_index[0]], num_classes=4)
        yield batch_features, batch_labels


def train(num_classes, x_train, y_train, x_test, y_test):
    # Hyperparameters
    num_tags = num_classes
    epochs = 50 # sampleCNN used 1000 epochs
    batch_size = 23
    lr = 0.01
    lrdecay = 1e-6
    conv_window_size = 3
    sample_length = 31104
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
    MP9 = MaxPooling1D(pool_size=2)(activ9)

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

    model.fit_generator(
        train_generator_from_song_list(x_train, y_train, batch_size),
        validation_data=test_generator_from_song_list(x_test, y_test, batch_size),
        validation_steps=len(x_test)/batch_size,
        callbacks=callbacks_list,
        verbose=1,
        epochs=epochs,
        steps_per_epoch=len(x_train)/batch_size * 3, # * 3 to see each song 3 times, gives data augmentation a good chance to work on each song in each epoch
        )
    return


if __name__ == '__main__':
    x_train, y_train = build_song_list(parent_dir=TRAIN_SONGS_DIR)
    x_test, y_test = build_song_list(parent_dir=TEST_SONGS_DIR)
    train(4, x_train, y_train, x_test, y_test)

    # validation data can also be generated using generator
    # just have a separate generator for validation data w/o data augmentation