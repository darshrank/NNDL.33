# NNDL.33

CoughSense is a lightweight cough detection project built around two approaches:
an MFCC-based random forest baseline and two CNN models prepared for Edge Impulse.
The goal is binary classification of cough vs. non-cough audio.

## Project Structure

- `src/preprocessing/mfcc_feature_extraction.py` extracts MFCC features from audio files.
- `src/models/model_1_random_forest_baseline.py` trains a scikit-learn random forest on MFCC features.
- `src/models/model_2_initial_cnn.py` defines the first CNN architecture used for Edge Impulse training.
- `src/models/model_3_final_optimized_cnn.py` defines the optimized CNN with batch normalization and class weights.
- `data/` contains the training and test audio files used by the scripts.
- `data/data_description.md` documents the Coswara dataset mapping used in this project.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The project depends on:

- `numpy`
- `librosa`
- `scikit-learn`
- `tensorflow`

## Dataset

The project uses a modified version of the Coswara respiratory audio dataset.
Audio is organized into two classes:

- `cough`
- `non_cough`

The baseline script expects the following folder layout:

```text
data/
	training/
		cough*.wav
		non_cough*.wav
	testing/
		cough*.wav
		non_cough*.wav
```

Files that do not end in `.wav` or do not start with `cough` or `non_cough` are ignored by the loader.

## Feature Extraction

`src/preprocessing/mfcc_feature_extraction.py` loads each audio file with `librosa`, pads or trims it to 3 seconds at 16 kHz, and extracts 13 MFCCs. The resulting feature vector is the concatenation of the MFCC means and standard deviations, producing 26 features per file.

## Models

### Random Forest Baseline

`src/models/model_1_random_forest_baseline.py`:

- loads MFCC features from the training and testing folders
- trains a `RandomForestClassifier` with 200 trees
- reports accuracy, confusion matrix, weighted precision/recall/F1, per-class F1, and ROC AUC

Run it with:

```bash
python src/models/model_1_random_forest_baseline.py
```

### Initial CNN

`src/models/model_2_initial_cnn.py` defines an initial 1D-to-2D reshaped CNN for Edge Impulse training. It uses Gaussian noise, two convolution blocks, a dense classifier head, and a softmax output layer.

### Final Optimized CNN

`src/models/model_3_final_optimized_cnn.py` defines the refined CNN architecture. Compared with the initial CNN, it adds batch normalization, deeper convolution blocks, dropout regularization, and class weights for imbalanced training.

Both CNN scripts are written to be used in the Edge Impulse training pipeline, where `input_length` and `classes` are supplied by the platform.

## Notes

- The preprocessing and baseline model scripts assume Python can import modules from `src/`.
- If you update the dataset layout or class labels, keep the filename prefixes and folder names aligned with the loader logic.
