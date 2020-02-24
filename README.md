A deep neural net to classify music by mood

## Dependencies:
Please install Miniconda and create a new virtual environment for this project. This will make package conflicts unlikely.

To install Miniconda please go to https://conda.io/miniconda.html

When it warns you about adding Miniconda to your path, make sure you add it!

You can replicate the current Mood Anaconda environment by cloning this repo, then running

```
conda env create -f environment.yml
```

## SpectrogramCNN

### Dependencies
Inside the `mood_algorithm/spectrogram-mood-classifier` directory you will find a `requirements.txt` file that lists all the necessary dependencies.

### Prediction
See the `main()` function in `/mood_algorithm/spectrogram-mood-classifier/predict.py` for example local and cloud predictions.
To predict on a bunch of songs using our current CloudML model, simply loop through all your songs, and pass their file names into the `cloud_predict()` function.

