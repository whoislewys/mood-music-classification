import os
import numpy as np
import librosa


def splitsongs(X, y, window=0.1, overlap=0.5):
    # split song samples into overlapping windows
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape * window)
    offset = int(chunk * (1. - overlap))

    # Split the song and create new ones on windows
    spsong = [X[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)


def extract_melspec(song, n_fft=1024, hop_length=512, log_scale=False):
    melspec = librosa.feature.melspectrogram(song, n_fft=n_fft, hop_length=hop_length)[:, :, np.newaxis]
    if log_scale:
        melspec = librosa.logamplitude(melspec**2, ref_power=1.0)  # log scale melspec
    return melspec


def sample_windows_to_melspecs(songs):
    melspecs = map(extract_melspec, songs)
    return np.array(list(melspecs))


def read_data(src_dir, classes_dict, num_samples, debug=True):
    # Empty array of dicts with the processed features from all files
    arr_specs = []
    arr_genres = []
    # Read files from the folders
    for class_key, _ in classes_dict.items():
        mood_folder = src_dir + '/' + class_key
        for root, subdirs, files in os.walk(mood_folder):
            for file in files:
                if not file.endswith('.mp3'):
                    print('skipping non mp3 file:', file)
                    continue
                # TODO: normalize
                # Read each audio file in cur folder
                file_name = os.path.join(mood_folder, file)
                if debug:
                    print('Reading {} samples from file: {}'.format(num_samples, file_name))
                samples, sr = librosa.load(file_name, sr=22050)
                samples = samples[:num_samples]
                print('samples shape: ', samples.shape);
                # Convert to dataset of melspectrograms
                windowed_samples, y = splitsongs(samples, classes_dict[class_key])
                # Convert to "spec" representation
                specs = sample_windows_to_melspecs(windowed_samples)
                arr_specs.extend(specs)  # extend adds the contents of specs list into the arr_specs list
                arr_genres.extend(y)
    return np.array(arr_specs), np.array(arr_genres)


if __name__ == '__main__':
    print('extracting data')
    # dataset_dir = '../../songs/training_data'
    # dataset_dir = '../../songs/smol_training_data'
    dataset_dir = '/home/lewys/dev/thesis/mood-music-classification/test_subset'
    song_samples = 660000  # roughly 30 secs at 22050 sr
    moods = {'angry-0': 0, 'happy-1': 1, 'sad-2': 2, 'romantic-3': 3}
    X, y = read_data(dataset_dir, moods, song_samples, debug=True)
    print('x shape: ', X.shape)
    print('y shape: ', y.shape)
    np.save('x_mood_dataset.npy', X)
    np.save('y_mood_dataset.npy', y)
