import numpy as np
import librosa
import tensorflow
from tensorflow.keras.utils import to_categorical
import tensorflow
import os

PATH_TO_GOOGLE_CREDENTIALS = r'C:\Users\lewys\PycharmProjects\mood_algorithm\local_data\moodmvp-15705c9d403a.json'
SAMPLE_RATE = 22050
OFFSET = 47
DURATION = 30
NUM_SAMPLES = 660000  # roughly 30 secs at 22050 sr

def extract_melspec(song, n_fft=1024, hop_length=512, log_scale=False):
    melspec = librosa.feature.melspectrogram(song, n_fft=n_fft, hop_length=hop_length)[:, :, np.newaxis]
    if log_scale:
        melspec = librosa.logamplitude(melspec**2, ref_power=1.0)  # log scale melspec
    return melspec


def to_melspectrogram(songs):
    melspecs = map(extract_melspec, songs)
    return np.array(list(melspecs))


def splitsongs(X, y, window=0.1, overlap=0.5):
    # split song samples into overlapping windows
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


def preprocess(file_name, num_samples=NUM_SAMPLES):
    # takes in song, returns windowed mel spectrograms
    print('loading file {}'.format(file_name))
    # samples, sr = librosa.load(file_name, sr=22050) # OG
    samples, sr = librosa.load(file_name, sr=SAMPLE_RATE, offset=OFFSET, duration=DURATION)
    samples = samples[:num_samples]
    # print(samples.shape)

    print('windowing file...')
    windowed_samples, y = splitsongs(samples, 0)
    X = to_melspectrogram(windowed_samples)
    y = to_categorical(y)
    return X, y


def calculate_mode_mood(mood_vals):
    '''
    Finds the most frequently predicted mood out of all the windows
    If there is a tie for number of windows predicted to be a certain mood,
    it will return -1 to indicate another function should be used for prediction.
    '''
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


def calculate_average_mood(mood_vals):
    '''
    Gets the max mood vals in each window & averages them. Could also average each column of the 5x4 window matrix, if for some reason we stop liking this function
    Only used if there is a tie for mode mood
    '''
    #
    avg_mood = np.mean(list(map(np.argmax, mood_vals)))
    print('average mood: ', avg_mood)
    pred_mood = np.round(avg_mood)
    return pred_mood


def get_overall_prediction(predictions_list):
    '''
    can be easily modifed to return a mood values vectorself.
    if mode mood is used, pick one of the mood vector at one of the max indices from calculate_mode_mood()
    if average mood is used, also return the avg_mood matrix from calculate_average_mood()
    '''
    most_frequent_max_mood = calculate_mode_mood(predictions_list)
    # print('most frequent max mood: ', most_frequent_max_mood)

    if most_frequent_max_mood == -1:
        # print('Tie for most frequent max. Calculating average...')
        avg_mood = calculate_average_mood(predictions_list)
        # print('avg mood: ', avg_mood)
        return avg_mood
    else:
        # print('Using mode mood')
        return most_frequent_max_mood


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
    print('submitting windows to cloudML...')
    for index, window in enumerate(windowed_X[-5:]):
        # print('submitting window #{}'.format(index))
        reshaped_window = np.reshape(window, (1, window.shape[0], window.shape[1], window.shape[2]))

        json_compatible_samples = reshaped_window.tolist()
        predict_object = json.dumps({'instances': json_compatible_samples})
        # write json to local storage before sending to cloudML to check size and stuff
        # with open('./instances.json', 'w+') as infile:
        #     infile.write(predict_dict)
        #print('Submission to CloudML shape: ', reshaped_window.shape)

        ml = discovery.build('ml', 'v1')
        name = 'projects/{}/models/{}'.format('moodmvp', 'Mood_MelspecCNN_bestof50')

        response = ml.projects().predict(
            name=name,
            body={'instances': json_compatible_samples}
        ).execute(num_retries=2)
        if 'error' in response:
            raise RuntimeError(response['error'])
        else:
            responses.extend(response['predictions'])
    return responses


def multi_cloud_predict(files):
    '''
    Given a list of song paths, returns the accuracy of predictions + prediction info for each song in the list.
    There are a couple split() calls to parse the correct label from the file path of the song.
    This has only been tested on MY machine, you will need to change the split call to pull the label out of your song's filepath.
    Args:
      files: a list of filepaths to songs
    Returns:
      accuracy of predictions to the list of songs
    '''
    correctly_classified_songs = 0
    total_songs = 0

    for fn in files:
        print('predicting for song: ', fn)
        cloudML_response = cloud_predict(fn)
        predictions_for_each_window = [mood_val['output'] for mood_val in cloudML_response]
        for i, pred in enumerate(predictions_for_each_window):
            print('prediction for window #{}: {}'.format(i, pred))

        prediction = get_overall_prediction(predictions_for_each_window) # mood predictions for each window
        print('Overall mood prediction: ', prediction)
        real_label = int(fn.split('\\')[7].split('-')[1])
        print('labeled mood: ', real_label)
        if prediction == real_label:
            correctly_classified_songs += 1
        total_songs += 1

    # calculate accuracy
    print('Total songs ', total_songs)
    print('Correctly classified songs ', correctly_classified_songs)
    acc = correctly_classified_songs/total_songs * 100
    print('Accuracy: {}%'.format(acc))
    print()
    return acc


def main():
    # file_name = r'C:\Users\lewys\Downloads\songs\awol.mp3'
    # local predict
    # model_name = os.path.join(os.getcwd(), '2D_mood_cnn.h5')
    # local_predict(file_name, model_name)

    # Cloud Predict
    file_path = r'../../data/Dinner at my Place.mp3'
    cloudML_response = cloud_predict(file_path)

    predictions_for_each_window = [mood_val['output'] for mood_val in cloudML_response]
    for i, pred in enumerate(predictions_for_each_window):
        print('prediction for window #{}: {}'.format(i, pred))

    pred = get_overall_prediction(predictions_for_each_window)
    print('Overall mood prediction: ', pred)


if __name__ == '__main__':
    main()
