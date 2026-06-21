import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


MFCC_COLUMNS = [
    "mean_MFCC_0th_coef",
    "mean_MFCC_1st_coef",
    "mean_MFCC_2nd_coef",
    "mean_MFCC_3rd_coef",
    "mean_MFCC_4th_coef",
    "mean_MFCC_5th_coef",
    "mean_MFCC_6th_coef",
    "mean_MFCC_7th_coef",
    "mean_MFCC_8th_coef",
    "mean_MFCC_9th_coef",
    "mean_MFCC_10th_coef",
    "mean_MFCC_11th_coef",
    "mean_MFCC_12th_coef",
]


def load_prepared_data(path="data/pd_speech_features.csv"):
    df = pd.read_csv(path, header=1)

    X = pd.DataFrame(
        {
            "pitch_mean": 1.0 / df["meanPeriodPulses"],
            "jitter": df["locPctJitter"],
            "shimmer": df["locShimmer"],
            "hnr_mean": df["meanHarmToNoiseHarmonicity"],
        }
    )

    for column in MFCC_COLUMNS:
        X[column] = df[column]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = df["class"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

    return train_test_split(
        X_balanced,
        y_balanced,
        test_size=0.2,
        random_state=42,
        stratify=y_balanced,
    )


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_prepared_data()
    print("Training and validation sets prepared.")
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("Training class distribution:", dict(pd.Series(y_train).value_counts().sort_index()))
