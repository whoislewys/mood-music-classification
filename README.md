A deep neural net to classify music by mood

## Dependencies:
Please install Miniconda3 to create a virtual environment for this project. This will make package conflicts unlikely.

To install Miniconda please go to https://conda.io/miniconda.html

> If it warns you about adding Miniconda to your path, make sure you add it!

You can replicate the current Mood Anaconda environment by cloning this repo, then running

```
conda env create -f environment.yml
```

## Running the live demo
Make sure you have follow the Dependencies instructions above.

In the top level of this repo, simply do
* `jupyter notebook`.
* Click on the `notebooks` dir
* Open `livedemo.ipynb`

And have fun!

## SpectrogramCNN

### Dependencies
Inside the `mood_algorithm/spectrogram-mood-classifier` directory you will find a `requirements.txt` file that lists all the necessary dependencies.

### Prediction
See the `main()` function in `/mood_algorithm/spectrogram-mood-classifier/predict.py` for example local and cloud predictions.
To predict on a bunch of songs using our current CloudML model, simply loop through all your songs, and pass their file names into the `cloud_predict()` function.

