import pyrubberband as pyrb
# Q: Why the aren't you using librosa's built-in stretch and pitch_shift?
# A: Try it, you'll see librosa's smears transients much more than pyrb. Greatly affects the feel of the track

def pitch_shift(insamples, sr, semitones):
    # Suggested semitone shifts: -2 -1 1 2
    # Anything more may be too destructive to the mood of a song
    #X, sr = librosa.load(insong, sr=sr)
    shifted_song = pyrb.pitch_shift(insamples, sr, semitones)
    return shifted_song


# def librosa_pitch_shift(insong, semitones):
#     y, sr = librosa.load(insong)
#     shifted_song = librosa.effects.pitch_shift(y, sr, semitones)
#     return shifted_song


def time_stretch(insamples, sr, rate):
    # Suggested stretch rates: 0:93, 1:07. Corresponds to roughly +10 BPM and -10 BPM respectively
    # The sound data augmentation paper I read used these values 0:81; 0:93; 1:07; 1:23 which positively impacted classification rates of street music vs other types of sounds
    stretched_song = pyrb.time_stretch(insamples, sr, rate)
    return stretched_song


# def librosa_time_stretch(insong, rate):
#     y, sr = librosa.load(insong)
#     stretched_song = librosa.effects.time_stretch(y, rate)
#     return stretched_song


def dynamic_range_compression(insong, dynamic_range):
    # Use compressor like loudnorm (can it be used on samples?) to reduce dynamic range of audio. Gives very different samples while easily retaining mood
    print('Dyanmic Range Compression is not yet implemented!')
    pass


if __name__ == '__main__':
    # save to .wav
    # librosa.output.write_wav(path=, y=, sr=,)
    pass
