import numpy as np
import librosa
import tensorflow
from tensorflow.keras.utils import to_categorical
# import keras
# from keras.utils import to_categorical
import tensorflow
import os

PATH_TO_GOOGLE_CREDENTIALS = r'C:\Users\lewys\PycharmProjects\mood_algorithm\local_data\google_ml_credentials.json'


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


def preprocess(file_name):
    # takes in song, returns windowed mel spectrograms
    # print('loading file {} with offset {}'.format(file_name, OFFSET))
    # samples, sr = librosa.load(file_name, sr=22050) # OG
    samples, sr = librosa.load(file_name, sr=22050, offset=43, duration=27)
    samples = samples[:590490] # make sampled song evenly divisble by 59049
    # print(samples.shape)

    print('windowing file...')
    X, y = splitsong(samples, sr, label=0, chunk_size_samples=59049)
    y = to_categorical(y)
    return X, y


def calculate_average_mood(mood_vals):
    avg_mood = np.mean(list(map(np.argmax, mood_vals)))
    print('average mood: ', avg_mood)
    pred_mood = np.round(avg_mood)
    return pred_mood


def calculate_mode_mood(mood_vals):
    # angry happy sad sexy
    mood_counts = [0, 0, 0, 0]
    # get the max index for each prediction window, use it to update the mood_counts array
    for window_vals in mood_vals:
        pred_for_window = np.argmax(window_vals)
        if(pred_for_window == 0):
            mood_counts[0] += 1
        elif(pred_for_window == 1):
            mood_counts[1] += 1
        elif(pred_for_window == 2):
            mood_counts[2] += 1
        elif(pred_for_window == 3):
            mood_counts[3] += 1
        else: # just in case
            print('ERROR. Got a max that is not between 0-3: ', pred_for_window)
            exit()

    max_mood_count = max(mood_counts)
    max_indices = [i for i, mood_count in enumerate(mood_counts) if mood_count == max_mood_count]
    if len(max_indices) > 1:
        # there is a tie for max values. return -1 to indicate that
        # mean mood should be used for overall prediction instead
        return -1
    else:
        # if there is no tie for most frequent max, simply return the index
        return max_indices[0]


def local_predict(file_name, model_name):
    X, y = preprocess(file_name)
    X = X[0]
    X = np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))

    print('loading model: ', model_name)
    model = tensorflow.keras.models.load_model(model_name)
    mood_vals = model.predict(X)

    print('mood vals for each song window:\n', mood_vals)
    return mood_vals


def cloud_predict(song_path):
    from oauth2client.client import GoogleCredentials
    from googleapiclient import discovery
    from googleapiclient import errors
    import json

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = PATH_TO_GOOGLE_CREDENTIALS
    credentials = GoogleCredentials.get_application_default()
    # print(credentials)
    responses = []
    windowed_X, y = preprocess(song_path)

    for index, window in enumerate(windowed_X[-5:]):
        print('submitting window #{}'.format(index))
        reshaped_window = np.reshape(window, (1, window.shape[0], 1))
        print('shape of reshaped window: ', reshaped_window.shape)

        json_compatible_samples = reshaped_window.tolist()
        predict_object = json.dumps({'instances': json_compatible_samples})
        # write json to local storage before sending to cloudML to check size and stuff
        # with open('./instances.json', 'w+') as infile:
        #     infile.write(predict_dict)
        #print('Submission to CloudML shape: ', reshaped_window.shape)

        ml = discovery.build('ml', 'v1')
        name = 'projects/{}/models/{}'.format('mood-algorithm', 'Mood_SampleCNN_big')

        response = ml.projects().predict(
            name=name,
            body={'instances': json_compatible_samples}
        ).execute(num_retries=2)
        if 'error' in response:
            raise RuntimeError(response['error'])
        else:
            responses.extend(response['predictions'])
    return responses


def get_overall_prediction(predictions_list):
    most_frequent_max_mood = calculate_mode_mood(predictions_list)
    # print('most frequent max mood: ', most_frequent_max_mood)

    if most_frequent_max_mood == -1:
        print('Tie for most frequent max. Calculating average...')
        avg_mood = calculate_average_mood(predictions_list)
        # print('avg mood: ', avg_mood)
        return avg_mood
    else:
        return most_frequent_max_mood


def main():
    file_name = r'C:\Users\lewys\Downloads\songs\awol.mp3'
    # local predict
    # model_name = os.path.join(os.getcwd(), '2D_mood_cnn.h5')
    # local_predict(file_name, model_name)

    # multi cloud predict
    files = [
      r'C:\Users\lewys\PycharmProjects\mood_algorithm\songs\submission_songs\happy-1\Warm Knightz  Ft. SoFlo Reserved, Codeine Kobe (Prod. B.Young)-331026203-13LUFSnorm.mp3',
      r'C:\Users\lewys\PycharmProjects\mood_algorithm\songs\submission_songs\happy-1\TOKYODREAM-390697497-13LUFSnorm.mp3',
      r'C:\Users\lewys\PycharmProjects\mood_algorithm\songs\submission_songs\happy-1\Mingled Radical Thoughts-297332371-13LUFSnorm.mp3',
      r'C:\Users\lewys\PycharmProjects\mood_algorithm\songs\submission_songs\happy-1\Here I Am (Prod. RJ)-330941711-13LUFSnorm.mp3',
      r'C:\Users\lewys\PycharmProjects\mood_algorithm\songs\submission_songs\happy-1\Hits-396735915-13LUFSnorm.mp3'
    ]
    for fn in files:
        print('predicting for song: ', fn)
        cloudML_response = cloud_predict(fn)
        predictions_for_each_window = [mood_val['output'] for mood_val in cloudML_response]
        for i, pred in enumerate(predictions_for_each_window):
            print('prediction for window #{}: {}'.format(i, pred))

        pred = get_overall_prediction(predictions_for_each_window) # mood predictions for each window
        print('\nOverall mood prediction: ', pred)
        print('actual mood: ', fn.split('\\')[7])

    # cloud predict
    # cloudML_response = cloud_predict(file_name)
    #
    # predictions_for_each_window = [mood_val['output'] for mood_val in cloudML_response]
    # for i, pred in enumerate(predictions_for_each_window):
    #     print('prediction for window #{}: {}'.format(i, pred))
    #
    # pred = get_overall_prediction(predictions_for_each_window) # mood predictions for each window
    # print('\nOverall mood prediction: ', pred)


if __name__ == '__main__':
    main()
