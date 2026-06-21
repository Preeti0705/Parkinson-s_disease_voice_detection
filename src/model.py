import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/pd_speech_features.csv"
MODEL_PATH = "parkinsons_rf_model.pkl"


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


def load_training_data():
    """Build a 17-feature table that matches src.extract_features.extract_features."""
    df = pd.read_csv(DATA_PATH, header=1)

    feature_table = pd.DataFrame(
        {
            "pitch_mean": 1.0 / df["meanPeriodPulses"],
            "jitter": df["locPctJitter"],
            "shimmer": df["locShimmer"],
            "hnr_mean": df["meanHarmToNoiseHarmonicity"],
        }
    )

    for column in MFCC_COLUMNS:
        feature_table[column] = df[column]

    feature_table = feature_table.replace([np.inf, -np.inf], np.nan)
    feature_table = feature_table.fillna(feature_table.median(numeric_only=True))

    X = feature_table.to_numpy(dtype=float)
    y = df["class"].astype(int).to_numpy()
    return X, y


def train_model():
    X, y = load_training_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )

    print("Starting Grid Search...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val)

    print("\n--- Training Results ---")
    print("Feature count:", X.shape[1])
    print("Class distribution:", dict(pd.Series(y).value_counts().sort_index()))
    print("Best hyperparameters found:", grid.best_params_)
    print("Cross-validation accuracy: {:.4f}".format(grid.best_score_))
    print("Validation accuracy: {:.4f}".format(accuracy_score(y_val, y_pred)))
    print("\nValidation classification report:")
    print(classification_report(y_val, y_pred))

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel pipeline successfully saved as '{MODEL_PATH}'")


if __name__ == "__main__":
    train_model()
