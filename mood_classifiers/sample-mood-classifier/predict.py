"""
Best performers so far: SampleCNN_KAIST_31104_12khz_off57-NORM-trainwsubmissions-testdata-binary-sigmoid-checkpoint.h5
    80% acc on test data (never seen before)

"""
import os
import glob
import numpy as np
import pandas as pd
import librosa
import json
from extract_samples import normalize
import keras
from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
OFFSET = 57.0


def extract_samples(file_name, offset):
    sr = 12000
    segment_length = 31104
    duration = 2.592 # 31104 samples @ 12000 khz
    X, sr = librosa.load(file_name, sr=sr, mono=True, offset=offset, duration=duration)
    if X.shape[0] > segment_length:
        X = X[:segment_length]
    return X


def get_song_batch(songs_dir, batch_size):
    batch = None
    for counter, song in enumerate(songs_dir):
        if counter < batch_size:
            song_samples = extract_samples(song, 57.0)
            print('Current song: ', song)
            if batch is None:
                batch = song_samples
            else:
                batch = np.vstack([batch, song_samples])
        else:
            break
    batch.shape = (batch.shape[0], batch.shape[1], 1)
    print('batch shape: ', batch.shape)
    return batch


def single_prediction_mood_vals(song_path, model):
    # TODO: Try to get more accurate prediction try slide extract samples offset around to get a few diff portions of the song and average all predictions
    X = extract_samples(song_path, offset=57.0)
    X.shape = (1, len(X), 1)
    prediction = model.predict(X)
    return prediction


def get_label_from_mood_values(song_path, mood_values):
    song_name = ''.join(song_path.split('\\')[-1:])
    prediction_string = 'Mood values: {}\n'.format(mood_values)
    mood_prediction = np.argmax(mood_values)
    if mood_prediction == 0:
        prediction_string += '{} is predicted to be: {}'.format(song_name, 'angry-0 >:(')
    elif mood_prediction == 1:
        prediction_string += '{} is predicted to be: {}'.format(song_name, 'happy-1 :)')
    elif mood_prediction == 2:
        prediction_string += '{} is predicted to be: {}'.format(song_name, 'sad-2 :(')
    elif mood_prediction == 3:
        prediction_string += '{} is predicted to be: {}'.format(song_name, 'romantic-3 <3')
    return prediction_string


def to_savedmodel(model, export_path):
    """Convert the Keras HDF5 model into TensorFlow SavedModel."""

    print('export_path: ', export_path)
    builder = saved_model_builder.SavedModelBuilder(export_path)
	# model went into production with this signature:
    # signature = predict_signature_def(inputs={'input': model.inputs[0]}, outputs={'income': model.outputs[0]})
	# i think it should've been this signature (look at inputs): 
	signature = predict_signature_def(inputs={'input': model.input}, outputs={'income': model.outputs[0]})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={
                # may need to add another signature for prediction
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature}
        )
        builder.save()


def batch_prediction_from_directory(dir, model, do_normalization, savelog=True):
    logfile = os.path.join(dir, 'predictions.json')
    if savelog == True:
        if os.path.exists(logfile):
            os.remove(logfile)

    for song in glob.glob(os.path.join(dir, '*.mp3')):
        song_name = ''.join(song.split('\\')[-1:])
        if do_normalization == True:
            norm_song = normalize(song)
        else:
            norm_song = song
        samples = extract_samples(norm_song, OFFSET)
        samples = np.reshape(samples, (1, len(samples), 1))
        mood_values = model.predict(samples)
        mood_prediction = np.argmax(mood_values)

        if savelog == True:
            # logging
            with open(logfile, 'a') as fp:
                data = {'song_name': song_name,
                        #'samples': samples.tolist(),
                        'mood_values': mood_values.tolist(),
                        'mood_prediction': int(mood_prediction)
                        }
                json_data = json.dumps(data)
                fp.write(json_data)
        get_label_from_mood_values(song, mood_values)

    return print('Your log is at {}'.format(logfile))


def multi_song_prediction_from_file(file_path, model):
    # TODO: add in save_log version?
    # if (save_log == True):
    #     with open('multi_song_prediction_{}_model_{}_file'.format(model_name, file_path)):
    total_songs = 0
    correctly_classified_songs = 0
    msgpack_data = pd.read_msgpack(file_path)
    for counter, song in enumerate(msgpack_data):
        song_name = ''.join(song['song'].split('\\')[-1:])
        samples = song['samples']
        samples = np.reshape(samples,(1,len(samples),1))
        mood_values = model.predict(samples)
        real_label = int(song['mood'].split('-')[1])
        prediction_label = get_label_from_mood_values(song_name, mood_values)
        print(prediction_label)
        print('Real label: {}\n'.format(song['mood']))
        # gather counts for accuracy calculation
        prediction = np.argmax(mood_values)
        if prediction == real_label:
            correctly_classified_songs += 1
        total_songs += 1

    # calculate accuracy
    print('Total songs ', total_songs)
    print('Correctly classified songs ', correctly_classified_songs)
    acc = correctly_classified_songs/total_songs
    print('Accuracy: {}%'.format(acc))
    return acc


def cloudML_predict(song_path):
    # POST https://ml.googleapis.com/v1/projects/my-project/models/my-model:predict
    # r = requests.post('https://ml.googleapis.com/v1/projects/mood-algorithm/models/Mood_SampleCNN:predict', json=predict_json)

    from oauth2client.client import GoogleCredentials
    from googleapiclient import discovery
    from googleapiclient import errors
    # pip install google-cloud 0.33.1
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\lewys\PycharmProjects\mood_algorithm\local_data\google_ml_credentials.json'
    credentials = GoogleCredentials.get_application_default()
    print(credentials)

    samples = extract_samples(song_path, 57)
    samples.shape = (1, len(samples), 1)
    json_compatible_samples = samples.tolist()
    predict_dict = json.dumps({'instances': json_compatible_samples})
    print('Submission to CloudML: ', predict_dict)

    ml = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format('mood-algorithm', 'Mood_SampleCNN')

    response = ml.projects().predict(
        name=name,
        body={'instances': json_compatible_samples}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])
    return response['predictions']


if __name__ == '__main__':
    # model_name = 'SampleCNN_KAIST_31104_12khz_off57-NORM-trainwsubmissions-testdata-binary-sigmoid-checkpoint.h5' # current production
    # model_name = 'SampleCNN_KAIST_31104_12khz_off57-NORM-testdata-binary-sigmoid-AFTER50.h5'
    model_name = 'SampleCNN_KAIST_31104_12khz_off57-NORM-testdata-binary-sigmoid-checkpoint-data-aug.h5'

    model = keras.models.load_model(os.path.join(os.pardir, 'local_data', model_name))
    print('Model {} loaded!'.format(model_name))

    ### turn keras model into tf graph and save it to upload to google cloud ML engine ###
    # to_savedmodel(model, export_path=r'C:\Users\lewys\PycharmProjects\mood_algorithm\gcloud_graphs')

    ### cloudML prediction ###
    # song_path = os.path.join(os.pardir, 'local_data', 'songs', 'lovesong.mp3')
    # print('cloud predicting on song path: ', song_path)
    # cloudML_prediction = cloudML_predict(song_path)
    # print('CloudML prediction: ', cloudML_prediction)

    ### single prediction and labeling ###
    # song_path = os.path.join(os.getcwd(), 'data', 'dinner-at-my-place.mp3')
    # mood_values = single_prediction_mood_vals(song_path, model)
    # prediction_label = get_label_from_mood_values(song_path, mood_values)
    # print(prediction_label)

    ### batch prediction from file with ACCURACY calculation###
    #file_name = 'submissions_samples_31104_12khz_off15-NORM.msg'
    #file_name = 'submission_samples_jun4_31104_12khz_off57-NORM.msg'
    #file_name = 'test_samples_jun4_31104_12khz_off57-NORM.msg'
    #file_path = os.path.join(os.pardir, 'local_data', file_name)
    #multi_song_prediction_from_file(file_path, model)

    ### comparison of batch prediction to multiple single prediction ###
    # angry_subs = os.path.join(os.getcwd(), r'songs\submission_songs\angry-0')
    # angry_subs = glob.glob(os.path.join(angry_subs, '*.mp3'))
    # print(angry_subs)
    # song_batch = get_song_batch(angry_subs, 3)
    # batch_preds = model.predict(song_batch, batch_size=3)
    # print('Batch predictions: ', batch_preds)
    # print()
    #
    # for counter, song_path in enumerate(angry_subs):
    #     if counter < 3:
    #         mood_values = single_prediction_mood_vals(song_path, model)
    #         print('vals for {}\n{}'.format(song_path, mood_values))
    #         print()

    ### batch prediction from directory ###
    predict_path = os.path.join(os.pardir, 'local_data', 'songs')  # a path with mp3s
    batch_prediction_from_directory(predict_path, model=model, do_normalization=False)