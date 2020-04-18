import os
import librosa
import numpy as np

def splitsong(X, sr, label, chunk_size_samples, overlap_amt=0.5):
    '''
    split (30sec) audio into 50% overlapping (3sec) windows
    '''
    # turn the chunk size in secs to the portion of the total song you want your chunks to be
    # should be 0.1 if incoming X is 30 secs of audio
    song_windows = []
    song_labels = []

    chunk_ratio = chunk_size_samples / X.shape[0]
    forward_hop = int(X.shape[0] * chunk_ratio * overlap_amt)
    for i in range(0, X.shape[0] - chunk_size_samples, forward_hop):
        song_chunk = X[i : i + chunk_size_samples]
        if len(song_chunk) != chunk_size_samples:
            print('error! song chunk is not {} samples'.format(chunk_size_samples))
            exit()
        else:
            # if length of chunk is good reshape it to fit into Conv1D layer
            song_chunk = np.reshape(song_chunk, (song_chunk.shape[0], 1))
            song_windows.append(song_chunk)
            song_labels.append(label)
            # print('song chunk shape: ', song_chunk.shape)

    return np.asarray(song_windows), np.asarray(song_labels)


def read_data(src_dir, classes_dict, debug=True):
    song_snippets = []
    song_labels = []
    # Read files from the folders
    for class_key, _ in classes_dict.items():
        mood_folder = src_dir + '/' + class_key
        for root, subdirs, files in os.walk(mood_folder):
            for file in files:
                if not file.endswith('.mp3'):
                    # sometimes i was getting desktop.ini files. regex would be better if we needed to support other file types
                    print('skipping non mp3 file:', file)
                    continue
                # TODO: normalize
                # Read each audio file in cur folder
                file_name = os.path.join(mood_folder, file)
                if debug:
                    print('Reading file: {}'.format(file_name))
                samples, sr = librosa.load(file_name, sr=22050, offset=43, duration=27)
                samples = samples[:590490] # make sampled song evenly divisble by 59049
                # Convert 30sec song sample to 3sec overlapping windows of song samples
                windowed_samples, y = splitsong(samples, sr, label=classes_dict[class_key], chunk_size_samples=59049)
                # extend adds the contents of a list into another list
                # because windowed_samples is a list of lists, it will add each list in windowed_samples to the song_snippets list
                print('windowed samples shape: ', windowed_samples.shape)
                song_snippets.extend(windowed_samples)
                song_labels.extend(y)
    return np.array(song_snippets), np.array(song_labels)


if __name__ == '__main__':
    print('extracting data')
    # dataset_dir = '../../songs/smol_training_data'
    # dataset_dir = '../../songs/smol_training_data'
    dataset_dir = '/home/lewys/dev/thesis/mood-music-classification/test_subset'
    song_samples = 59049  # roughly 30 secs at 22050 sr
    moods = {'angry-0': 0, 'happy-1': 1, 'sad-2': 2, 'romantic-3': 3}
    X, y = read_data(dataset_dir, moods)
    print('saving X with shape: ', X.shape)
    np.save('x_mood_dataset_samples_22k_off43_dur59049.npy', X)
    np.save('y_mood_dataset_samples_22k_off43_dur59049.npy', y)
