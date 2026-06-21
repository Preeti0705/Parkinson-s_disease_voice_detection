# Parkinson's Disease Voice Detection - Project Walkthrough

This project is a voice-based Parkinson's disease risk screening prototype. It combines a machine learning model with a Streamlit web app so a user can record a short sustained vowel sound, extract acoustic voice features, and receive a predicted class with a confidence score.

Important note: this is a screening/demo system, not a medical diagnostic tool.

## Short Answer: Is The Claimed Technique Being Used?

Partially, yes.

The codebase currently uses or contains:

- Librosa for audio loading, resampling, trimming, and MFCC extraction.
- Praat-Parselmouth for clinical-style acoustic features such as pitch, jitter, shimmer, and harmonicity.
- StandardScaler in the training/preparation scripts.
- SMOTE in the training/preparation scripts.
- RandomForestClassifier as the saved model type.
- Joblib for saving/loading the model.
- Streamlit for the real-time web interface.
- `data/pd_speech_features.csv` exists in the project and contains the high-dimensional UCI Parkinson's speech feature dataset.

The current model/training path is now wired around `data/pd_speech_features.csv`:

- `dataprep.py` reads `data/pd_speech_features.csv`.
- `src/model.py` trains from `data/pd_speech_features.csv`.
- The saved model `parkinsons_rf_model.pkl` is a scaler + SMOTE + Random Forest pipeline.
- The saved model expects 17 input features, matching `src/extract_features.py`.
- The `pd_speech_features.csv` dataset has 755 columns when read with `header=1`, including the target column `class`.

So the project now matches the main pipeline story: local CSV data, StandardScaler, SMOTE, Random Forest, Joblib deployment, and 17 live audio features.

## Project Goal

The goal is to detect Parkinson's disease risk from voice data.

The expected user-facing flow is:

1. User opens the Streamlit app.
2. User records a short sustained vowel sound, usually "ah".
3. The app stores the recording temporarily as a `.wav` file.
4. Audio features are extracted from the recording.
5. Features are passed to a trained Random Forest model.
6. The model predicts:
   - `0` = Healthy
   - `1` = Parkinson's Disease
7. The app displays the prediction and confidence score.

## Repository Structure

```text
.
├── app.py
├── dataprep.py
├── data_analysis.ipynb
├── parkinsons_rf_model.pkl
├── requirements.txt
├── walkthrough.md
├── data/
│   ├── pd_speech_features.csv
│   └── pd_speech_features.rar
└── src/
    ├── audio_prepro.py
    ├── extract_features.py
    ├── model.py
    └── predict.py
```

## Architecture Overview

The project has three main layers:

```text
Data Layer
    data/pd_speech_features.csv
    Precomputed Parkinson's speech features and class labels

ML Pipeline Layer
    dataprep.py
    src/model.py
    StandardScaler -> SMOTE -> RandomForestClassifier -> Joblib model

Application Layer
    app.py
    Streamlit UI -> microphone recording -> feature extraction -> model prediction
```

## End-To-End Intended Pipeline

The intended machine learning pipeline is:

```text
Raw / precomputed voice data
        |
        v
Load dataset
        |
        v
Split features X and target y
        |
        v
Scale features with StandardScaler
        |
        v
Balance classes with SMOTE
        |
        v
Train Random Forest model
        |
        v
Save trained model with Joblib
        |
        v
Load model in Streamlit app
        |
        v
Record user audio
        |
        v
Extract audio features with Librosa + Praat
        |
        v
Predict Healthy / Parkinson's Risk
```

## Dataset: `data/pd_speech_features.csv`

This file is the local dataset you mentioned.

When read as:

```python
pd.read_csv("data/pd_speech_features.csv", header=1)
```

the dataset has:

- 756 rows
- 755 columns
- target column: `class`
- class distribution:
  - `1`: 564 samples
  - `0`: 192 samples

This is imbalanced because Parkinson's samples are much more common than healthy samples in the dataset.

That imbalance is why SMOTE is useful. SMOTE creates synthetic minority-class examples so the model does not become biased toward the majority class.

### Feature Groups In The CSV

The CSV contains many categories of voice features, including:

- Baseline vocal features
- Jitter features
- Shimmer features
- Harmonicity / noise features
- Intensity parameters
- Formant frequencies
- Bandwidth parameters
- Vocal fold features
- MFCC features
- Wavelet features
- TQWT features

The label column is:

```text
class
```

where:

```text
0 -> Healthy
1 -> Parkinson's Disease
```

## File-By-File Explanation

## `requirements.txt`

This file lists the Python packages needed to run the project.

Current important dependencies:

- `numpy`: numerical arrays and mathematical operations.
- `pandas`: reading and manipulating CSV datasets.
- `scikit-learn`: StandardScaler, RandomForestClassifier, GridSearchCV, train/test split.
- `imbalanced-learn`: SMOTE for handling class imbalance.
- `librosa`: audio loading, resampling, trimming, and MFCC extraction.
- `soundfile`: audio file support used by audio libraries.
- `joblib`: saving/loading trained machine learning models.
- `streamlit`: web app framework.
- `streamlit-mic-recorder`: browser microphone recorder component.
- `praat-parselmouth`: Python interface to Praat acoustic analysis.

## `app.py`

This is the main Streamlit web application.

Responsibilities:

1. Imports Streamlit, Joblib, base64, tempfile, OS utilities, and the microphone recorder.
2. Imports `extract_features` from `src.audio_prepro`.
3. Loads `parkinsons_rf_model.pkl` using Joblib.
4. Displays a microphone recording interface.
5. Receives recorded audio from the browser.
6. Writes the audio into a temporary `.wav` file.
7. Extracts features from the audio file.
8. Calls the trained model's `predict()` and `predict_proba()`.
9. Displays the predicted class and confidence score.
10. Deletes the temporary audio file.

### App Flow

```text
User records audio
        |
        v
streamlit_mic_recorder returns audio bytes
        |
        v
app.py saves bytes to a temporary .wav file
        |
        v
extract_features(temp_file)
        |
        v
model.predict(features)
        |
        v
model.predict_proba(features).max()
        |
        v
Streamlit displays result
```

### Important Current Issue In `app.py`

`streamlit_mic_recorder` usually returns raw bytes in `audio_data["bytes"]`. The current function name says `audio_base64_string`, and the code calls:

```python
base64.b64decode(audio_base64_string)
```

If the recorder already returns bytes, base64 decoding may be unnecessary or may fail depending on the exact component output. This should be tested during app execution.

## `src/audio_prepro.py`

This file currently contains:

- `preprocess_audio(audio_path, sr_target=44100)`
- an import that exposes `extract_features` from `src.extract_features`

### `preprocess_audio`

This function:

1. Loads audio using Librosa.
2. Keeps the original sample rate initially.
3. Resamples audio to 44,100 Hz if needed.
4. Normalizes the waveform.
5. Trims silence.
6. Returns the cleaned audio and target sample rate.

Current function:

```python
def preprocess_audio(audio_path, sr_target=44100):
    audio, sr = librosa.load(audio_path, sr=None)
    if sr != sr_target:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_target)
    audio = audio / np.max(np.abs(audio))
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    return audio_trimmed, sr_target
```

### Current Note

`preprocess_audio()` is defined but not currently used by `app.py` before feature extraction. The extractor loads the audio file directly.

## `src/extract_features.py`

This file extracts acoustic features from a recorded audio file.

It uses:

- Praat-Parselmouth for pitch, jitter, shimmer, and harmonicity.
- Librosa for MFCCs.
- NumPy for numerical aggregation.

### Current Extracted Features

The function currently returns 17 features:

- 1 pitch feature:
  - mean pitch
- 1 jitter feature:
  - local jitter
- 1 shimmer feature:
  - local shimmer
- 1 harmonicity feature:
  - mean harmonics-to-noise ratio
- 13 MFCC features:
  - mean of each MFCC coefficient

Total:

```text
1 + 1 + 1 + 1 + 13 = 17 features
```

### Current Feature Extraction Flow

```text
Audio file path
        |
        v
parselmouth.Sound(audio_path)
        |
        v
Pitch, jitter, shimmer, harmonicity
        |
        v
librosa.load(audio_path)
        |
        v
MFCC extraction
        |
        v
Combine into NumPy feature vector
```

### Current Compatibility

The saved model has been retrained to expect these same 17 features.

## `dataprep.py`

This is a simple data preparation script.

It is intended to:

1. Load a Parkinson's dataset.
2. Separate features and target label.
3. Scale features with StandardScaler.
4. Balance classes with SMOTE.
5. Split the balanced data into training and validation sets.

Current code reads:

```python
df = pd.read_csv("data/pd_speech_features.csv", header=1)
```

It then builds the same 17-feature layout used by the real-time extractor:

- derived pitch from `meanPeriodPulses`
- local jitter
- local shimmer
- mean harmonic-to-noise ratio
- 13 mean MFCC coefficients

## `src/model.py`

This is the main training script for the Random Forest model.

Current responsibilities:

1. Reads `data/pd_speech_features.csv`.
2. Builds the same 17-feature layout used by live audio prediction.
3. Splits data into training and validation sets.
4. Scales features using StandardScaler.
5. Applies SMOTE to balance the training data.
6. Defines a Random Forest classifier.
7. Uses GridSearchCV for hyperparameter tuning.
8. Saves the best scaler + SMOTE + model pipeline as:

```text
parkinsons_rf_model.pkl
```

### Current Training Flow

```text
data/pd_speech_features.csv
        |
        v
17 feature table + class label
        |
        v
train_test_split(...)
        |
        v
GridSearchCV(
    StandardScaler -> SMOTE -> RandomForestClassifier
)
        |
        v
joblib.dump(best_pipeline, "parkinsons_rf_model.pkl")
```

The scaler is saved inside the pipeline, so `app.py` can pass raw extracted features directly to the loaded model.

## `src/predict.py`

This file is currently empty.

It could be used later to centralize prediction logic so both the Streamlit app and any command-line/API interface can call the same function.

A good future structure would be:

```text
src/predict.py
    load_model()
    predict_audio_file(audio_path)
    predict_feature_vector(features)
```

## `parkinsons_rf_model.pkl`

This is the saved machine learning model.

It loads as:

```text
RandomForestClassifier
```

The model is now an imbalanced-learn `Pipeline` containing:

```text
StandardScaler -> SMOTE -> RandomForestClassifier
```

It expects exactly 17 numeric input features per sample, matching the live extractor.

## `data/pd_speech_features.rar`

This is likely the compressed source archive for `pd_speech_features.csv`.

It is not used directly by the Python code.

## `data_analysis.ipynb`

This notebook is currently empty.

It could be used for exploratory data analysis, such as:

- checking class balance
- visualizing feature distributions
- correlation analysis
- feature selection
- model comparison
- evaluation reports

## Machine Learning Concepts Used

## Librosa

Librosa is used for audio signal processing.

In this project, it helps with:

- loading `.wav` files
- resampling audio
- trimming silence
- extracting MFCC features

MFCCs are widely used in speech and audio classification because they summarize the shape of the vocal spectrum.

## Praat-Parselmouth

Praat is a well-known tool for phonetics and speech analysis. Parselmouth provides Python access to Praat.

In this project, it is used to calculate:

- pitch
- jitter
- shimmer
- harmonicity / HNR

These are clinically relevant voice features because Parkinson's disease can affect vocal stability, loudness, and periodicity.

## StandardScaler

StandardScaler transforms features so they have:

```text
mean = 0
standard deviation = 1
```

This matters because voice features can have very different numerical ranges. Scaling prevents large-range features from dominating model training.

## SMOTE

SMOTE stands for Synthetic Minority Over-sampling Technique.

It handles class imbalance by creating synthetic examples of the minority class. In this dataset, healthy samples are fewer than Parkinson's samples:

```text
Healthy: 192
Parkinson's: 564
```

Without balancing, the model may learn to predict the majority class too often.

## Random Forest Classifier

Random Forest is an ensemble of decision trees.

It is useful here because:

- it works well on tabular numerical features
- it can model nonlinear relationships
- it is relatively robust
- it provides class probabilities through `predict_proba()`

## Joblib

Joblib is used to persist trained machine learning objects to disk.

The app loads:

```python
model = joblib.load("parkinsons_rf_model.pkl")
```

and then uses:

```python
model.predict(features)
model.predict_proba(features)
```

## Streamlit

Streamlit turns the model into a web app.

The app provides:

- page title and instructions
- microphone recording
- prediction trigger
- confidence metric
- audio playback
- medical disclaimer

## Current Reality vs Ideal Project Pipeline

## Current Reality

```text
app.py
    loads parkinsons_rf_model.pkl
    records microphone audio
    extracts 17 features
    sends features to a pipeline expecting 17 features

src/model.py
    trains from data/pd_speech_features.csv
    uses StandardScaler and SMOTE
    saves the complete pipeline

data/pd_speech_features.csv
    exists locally
    is used by the training scripts
```

## Ideal Consistent Pipeline

```text
data/pd_speech_features.csv
        |
        v
train_pipeline.py
        |
        v
Drop target column class
Drop non-signal ID columns if needed
        |
        v
StandardScaler
        |
        v
SMOTE
        |
        v
RandomForestClassifier
        |
        v
Save complete model pipeline with Joblib
        |
        v
Streamlit app loads the same pipeline
        |
        v
Real-time features match training features exactly
```

The most important ML rule is:

```text
The features used at prediction time must match the features used at training time.
```

Currently, that rule is not fully satisfied.

## What To Say In A Project Explanation

A technically accurate version of the project description would be:

> This project implements a Parkinson's disease voice screening prototype using acoustic voice features and a Random Forest classifier. The training pipeline applies feature scaling with StandardScaler and class balancing with SMOTE to handle imbalanced medical speech data. The Streamlit interface records a sustained vowel sample, extracts clinically relevant audio features using Librosa and Praat-Parselmouth, loads the trained model with Joblib, and displays a risk label with confidence.

If you are specifically claiming the deployed model uses `data/pd_speech_features.csv`, then the training script should be updated to train from that CSV and the saved model should be regenerated.

## Recommended Fixes Before Final Presentation

1. Update the training script to use `data/pd_speech_features.csv`.
2. Save a complete sklearn/imblearn pipeline that includes scaling and the Random Forest model.
3. Make real-time feature extraction produce the same feature columns used during training.
4. Save feature column names used during training.
5. Keep the model/input contract fixed at 17 features unless both training and extraction are changed together.
6. Test `streamlit_mic_recorder` output to confirm whether base64 decoding is needed.
7. Retrain the model with the same scikit-learn version used in deployment.
8. Clean up unused or outdated files such as the old `dataprep.py` path.

## How To Run

Install dependencies:

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
venv\Scripts\streamlit.exe run app.py
```

Train the current model script:

```powershell
venv\Scripts\python.exe src\model.py
```

Note: the current model script trains from `data/pd_speech_features.csv`.

## Final Summary

This codebase is a strong prototype for Parkinson's voice screening. It contains the right major building blocks: Streamlit, Librosa, Praat-Parselmouth, StandardScaler, SMOTE, Random Forest, and Joblib.

The main thing to fix is consistency. The dataset, training script, saved model, scaler, and real-time feature extractor must all agree on the exact same feature set. Once that is fixed, the project story and the implementation will line up cleanly.
