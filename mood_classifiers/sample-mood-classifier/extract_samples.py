"""
This program will extract audio samples to msgpack
Author: Luis Gomez
Email : luis.moodindustries@gmail.com
"""
import glob
import os
import sys
import time
import random
from itertools import chain
import subprocess
from concurrent.futures import ProcessPoolExecutor
import librosa
import math
import pandas as pd
import json

MOOD_DIR = os.pardir
TRAIN_SONGS_DIR = os.path.join(MOOD_DIR, 'songs', 'training_data')
TEST_SONGS_DIR = os.path.join(MOOD_DIR, 'songs', 'testing_data')
SUBMISSION_SONGS_DIR = os.path.join(MOOD_DIR, 'songs', 'submission_songs')
TRAIN_SUB_DIRS = os.listdir(TRAIN_SONGS_DIR)
TEST_SUB_DIRS = os.listdir(TEST_SONGS_DIR)
SUBMISSION_SUB_DIRS = os.listdir(SUBMISSION_SONGS_DIR)
TRAIN_FILE = 'train_samples_jun4_31104_12khz_off57-NORM.msg'
TEST_FILE = 'test_samples_jun4_31104_12khz_off57-NORM.msg'
SUBMISSIONS_FILE = 'submission_samples_jun4_31104_12khz_off57-NORM.msg'
FILE_EXT = '*.mp3'


class Options:
    def __init__(self, sr, offset, duration):
        self.sr = sr
        self.offset = offset
        self.duration = duration
        self.samples_length = duration * sr


options = Options(sr=12000, offset=57.0, duration=2.592)


# extract features from a song
def extract_samples(song_path):
    print('Extracting features from: ', song_path)
    X, sr = librosa.load(song_path, sr=options.sr, offset=options.offset, duration=options.duration)
    if X.shape[0] > options.samples_length:
        X = X[:options.samples_length]
    return X


def extract_random_samples(song_path):
    # may not be more time efficient than serializing, because it has to resample every time it loads a song
    # print('Extracting features from: ', song_path)
    samples, sr = librosa.load(song_path, sr=options.sr)
    song_duration_secs = librosa.get_duration(samples, sr=options.sr) # returns duration in seconds
    last_possible_offset = song_duration_secs - options.duration # make sure to not out of bounds when sampling
    random_offset = math.floor(random.uniform(0, last_possible_offset) * 100) / 100 # do not round to two decimals; that can cause sampling to go OOB. instead, floor to two decimals
    X, sr = librosa.load(song_path, sr=options.sr, offset=random_offset, duration=options.duration)
    return X


def normalize_and_extract_samples(song_path):
    norm_song_path = normalize(song_path)
    print('Extracting features from: ', norm_song_path)
    # sr = 8000  # AUC .9031    20736 samples   2592 ms  .55 faster than 22050
    sr = 12000 # AUC .9033    43740 samples   2592 ms  .69 faster than 22050
    # sr = 20000 # AUC .9055    52488 samples   2624 ms  .79 faster than 22050
    # sr = 22050 # AUC .9055      59049 samples   2678 ms  baseline
    offset = 57.0
    # duration = 2.678  # 59050 samples @ 22050 khz
    duration = 2.592 # 31104 samples @ 12000 khz
    X, sr = librosa.load(norm_song_path, sr=sr, offset=offset, duration=duration)
    samples_length = 31104
    if X.shape[0] > samples_length:
        X = X[:samples_length]
    return X


def print_normalize_warning():
    print('WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING')
    print('This will REPLACE songs in the parse_audio directories with normalized versions. Do you wish to continue?')
    normalize_option = input('Y/N: ').lower()
    if normalize_option != 'y':
        exit()
    return


def normalize(insong):
    '''
    spotify uses replay gain to meaure audio. replaygain is an algorithm to measure the perceived playback loudness
    it is calculated by doing the RMS after applying an equal loudness contour
    note: i tested in reaper.
    Had a song with lufs -8. I applied -4 db of gain to the song. Reanalyzed. It showed lufs of -12
    '''
    outsong = insong.split('.mp3')[0] + '-13LUFSnorm.mp3'
    if os.path.exists(outsong):
        return outsong

    print('normalizing...')
    # Use ffprobe to get Stereo/Mono and Sample Rate information
    ffprobe_results = subprocess.check_output(['ffprobe', '-show_streams', '-of', 'json', '-loglevel', '0', insong])
    json_stream_info = json.loads(ffprobe_results.decode('utf-8'))
    input_sr = json_stream_info['streams'][0]['sample_rate']
    # channel_layout = json_stream_info['streams'][0]['channel_layout'] #Stereo/Mono

    # Get Integrated Loudness in LUFS by doing a loudnorm dry run
    loudnorm_dry_run = subprocess.Popen(['ffmpeg', '-i', insong, '-af', 'loudnorm=I=-14:TP=-1.0:LRA=9:print_format=json', '-f', 'null', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = loudnorm_dry_run.communicate()
    stderr = stderr.decode('utf-8').split('{')
    json_result = '{' + stderr[1][:-2]
    loudnorm_results_json = json.loads(json_result)
    measured_I = float(loudnorm_results_json['input_i'])
    #print('Input measured_I: ', measured_I)

    # Check audio volume in dB volumedetect instead of in LUFS with a loudnorm dry run
    # ffmpeg -i video.avi -af "volumedetect" -vn -sn -dn -f null /dev/null
    # Only tested on one song, but it was about .5 db lower than loudnorm estimate. may want to use in future its probably a lil faster
    # volume_detect_results = subprocess.check_output(['ffmpeg', '-i', insong, '-af', 'volumedetect', '-vn', '-sn', '-dn', '-f', 'null', 'NUL'])
    # volume_detect_results_str = volume_detect_results.decode('utf-8')
    # print(volume_detect_results_str)
    # print('^volume detect^')

    # Normalize audio to -14 LUFS using RMS volume
    target_volume = -13
    gain = target_volume - measured_I
    #print('applying {}dB of gain'.format(gain))
    subprocess.call(['ffmpeg', '-y', '-i', insong, '-af', 'volume={}dB'.format(gain), '-ar', input_sr, outsong])
    # TODO: implement a fix for clipping. or should i let it clip? idk yet
    os.remove(insong)
    return outsong

def parse_audio(parent_dir, sub_dirs, feature_file, do_normalization, file_ext=FILE_EXT):
    # if feature file already exists, return
    if do_normalization:
        print_normalize_warning()
    feature_file_exists = os.path.isfile(os.path.join(MOOD_DIR, feature_file))
    if feature_file_exists:
        print("\nFeature file {} already exists! Skipping that for now.\nIf you'd like to extract features again, please delete {}.".format(feature_file, feature_file))
        return False
    else:
        # access each class/tag folder
        for sub_dir in sub_dirs:
            # extract features from each song in a folder
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)): # glob returns all files matching a wildcard
                try:
                    print('Extracting {}'.format(fn))
                    if do_normalization:
                        normalized_fn = normalize(fn)
                        samples = extract_samples(normalized_fn)
                    else:
                        samples = extract_samples(fn)
                except Exception as e:
                    print("Error encountered while parsing file: ", fn)
                    continue
                if sys.platform == 'win32':
                    mood = fn.split('\\')[3]
                else:
                    mood = fn.split('/')[8]
                serialize_to_msgpack(fn, mood, samples, feature_file)
    return True


def parallel_parse_audio(parent_dir, sub_dirs, feature_file, do_normalization, file_ext=FILE_EXT):
    if do_normalization:
        print_normalize_warning()
    feature_file_exists = os.path.isfile(os.path.join(MOOD_DIR, feature_file))
    if feature_file_exists:
        print("\nFeature file {} already exists! Skipping that for now.\nIf you'd like to extract features again, please delete {}.".format(feature_file, feature_file))
        return False

    # TODO: make the subdirs clearer
    song_paths_glob = [glob.glob(os.path.join(parent_dir, sub_dir, file_ext)) for sub_dir in sub_dirs] # get a list of songs - one for each sub directory in the parent directory
    song_paths = list(chain.from_iterable(song_paths_glob))

    # TODO: Fix the EOF error
    with ProcessPoolExecutor(max_workers=4) as executor:
        if do_normalization:
            feature_extract_func = normalize_and_extract_samples
        else:
            feature_extract_func = extract_samples

        for song_path, song_samples in zip(song_paths, executor.map(feature_extract_func, song_paths)):
            if sys.platform == 'win32':
                mood = song_path.split('\\')[7]
            else:
                mood = song_path.split('/')[8]
            serialize_to_msgpack(song_path, mood, song_samples, feature_file)
    return True


def serialize_to_msgpack(fn, mood, samples, feature_file):
        series_samples = pd.Series(samples)
        data = {
                'song': fn,
                'mood': mood,
                'samples': series_samples
                }
        pd.to_msgpack(feature_file, data, append=True)
        return data


def main():
    print('main')
    # parse_audio(TRAIN_SONGS_DIR, TRAIN_SUB_DIRS, TRAIN_FILE, do_normalization=False)
    # parse_audio(TEST_SONGS_DIR, TEST_SUB_DIRS, TEST_FILE, do_normalization=False)
    # parse_audio(SUBMISSION_SONGS_DIR, SUBMISSION_SUB_DIRS, SUBMISSIONS_FILE, do_normalization=False)

    # Time Testing
    # concurrent_test_dir = r'C:\Users\lewys\PycharmProjects\mood_algorithm\songs\concurrent_songs'
    # concurrent_test_subdirs = os.listdir(concurrent_test_dir)
    # test_file0 = 'test_samples_31104_12khz_off57-NORM-concurrent.msg'
    # test_file1 = 'test1_samples_31104_12khz_off57-NORM-concurrent.msg'
    #
    # print('NO normalization')
    # begin = time.time()
    # parse_audio(concurrent_test_dir, concurrent_test_subdirs, test_file0, do_normalization=True)
    # end = time.time()
    # print('Normalize and extract songs single core: ', end - begin)
    # # END TIME w/o norm: 70.8829870223999
    # # END TIME w norm:
    #
    # print('YEAH normalization!')
    # begin = time.time()
    # parallel_parse_audio(concurrent_test_dir, concurrent_test_subdirs, test_file1, do_normalization=True)
    # end = time.time()
    # print('Normalize and extract songs multi core: ', end - begin)
    # # END TIME w/o norm:
    # # END TIME w norm:


if __name__ == "__main__":
    main()
