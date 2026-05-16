"""
Train a lightweight model on UCI raw inertial signals for real-time mobile prediction.
Features: mean, std, min, max, rms per channel × 9 channels = 45 features.
Saves: models/realtime_model.pkl  +  models/realtime_scaler.pkl
"""

import numpy as np
import pandas as pd
import joblib, os, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

BASE = "UCI HAR Dataset/"
CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

ACTIVITIES = {1:"WALKING", 2:"WALKING_UPSTAIRS", 3:"WALKING_DOWNSTAIRS",
              4:"SITTING", 5:"STANDING", 6:"LAYING"}


def load_inertial(split: str) -> np.ndarray:
    """Load all 9 channels → (N, 9, 128)."""
    folder = BASE + f"{split}/Inertial Signals/"
    arrays = []
    for ch in CHANNELS:
        path = folder + f"{ch}_{split}.txt"
        arr  = pd.read_csv(path, sep=r"\s+", header=None).values  # (N, 128)
        arrays.append(arr)
    return np.stack(arrays, axis=1)  # (N, 9, 128)


def extract_features(windows: np.ndarray) -> np.ndarray:
    """
    windows: (N, 9, 128)
    Returns: (N, 45) — mean, std, min, max, rms per channel
    """
    mean = windows.mean(axis=2)                                   # (N,9)
    std  = windows.std(axis=2)
    mn   = windows.min(axis=2)
    mx   = windows.max(axis=2)
    rms  = np.sqrt((windows ** 2).mean(axis=2))
    return np.concatenate([mean, std, mn, mx, rms], axis=1)       # (N,45)


def feature_names() -> list:
    names = []
    for stat in ["mean", "std", "min", "max", "rms"]:
        for ch in CHANNELS:
            names.append(f"{ch}_{stat}")
    return names


print("Loading raw inertial signals …")
X_train_raw = load_inertial("train")   # (7352, 9, 128)
X_test_raw  = load_inertial("test")    # (2947, 9, 128)

y_train = pd.read_csv(BASE + "train/y_train.txt", header=None).squeeze().values
y_test  = pd.read_csv(BASE + "test/y_test.txt",   header=None).squeeze().values

print(f"Train windows: {X_train_raw.shape}  |  Test windows: {X_test_raw.shape}")

print("Extracting features …")
X_train = extract_features(X_train_raw)  # (7352, 45)
X_test  = extract_features(X_test_raw)   # (2947, 45)

feat_cols = feature_names()
print(f"Feature matrix shape: {X_train.shape}")

print("Scaling …")
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("Training SVM (RBF, C=5) …  (may take ~30 s)")
svm = SVC(kernel="rbf", C=5, gamma="scale", probability=True, random_state=42)
svm.fit(X_train_sc, y_train)

y_pred = svm.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred,
      target_names=[ACTIVITIES[i] for i in range(1,7)]))

os.makedirs("models", exist_ok=True)
joblib.dump(svm,    "models/realtime_model.pkl")
joblib.dump(scaler, "models/realtime_scaler.pkl")
joblib.dump(feat_cols, "models/realtime_feature_names.pkl")
print("Saved → models/realtime_model.pkl  |  models/realtime_scaler.pkl")
