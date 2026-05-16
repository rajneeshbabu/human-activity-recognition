# 🏃 Human Activity Recognition

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge)](https://human-activity-recognition-6zwy99fqmtsp66mmprn6sn.streamlit.app/)
[![GitHub Pages](https://img.shields.io/badge/🌐_Project_Page-GitHub_Pages-6366f1?style=for-the-badge)](https://rajneeshbabu.github.io/human-activity-recognition/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)

Classify six daily activities from smartphone accelerometer and gyroscope data using machine learning. Built on the UCI HAR Dataset — recordings of 30 subjects wearing a Samsung Galaxy S II at the waist.

**🚀 Live Demo:** [human-activity-recognition-6zwy99fqmtsp66mmprn6sn.streamlit.app](https://human-activity-recognition-6zwy99fqmtsp66mmprn6sn.streamlit.app/)
> Open on your **phone** → tap **▶ Start** → move naturally → see real-time continuous activity prediction!

**🌐 Project Page:** [rajneeshbabu.github.io/human-activity-recognition](https://rajneeshbabu.github.io/human-activity-recognition/)

---

## 📊 Dataset

| Property | Value |
|---|---|
| Subjects | 30 volunteers, age 19–48 |
| Activities | WALKING · WALKING_UPSTAIRS · WALKING_DOWNSTAIRS · SITTING · STANDING · LAYING |
| Sensor | Samsung Galaxy S II (accelerometer + gyroscope) |
| Sampling rate | 50 Hz |
| Window | 2.56 s, 50% overlap (128 readings/window) |
| Pre-extracted features | 561 (time + frequency domain) |
| Raw inertial signals | 9 channels × 128 timesteps |
| Train / Test split | 7,352 / 2,947 samples (70/30 split by subject) |

**Download dataset:** [UCI ML Repository — HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
Extract and place the `UCI HAR Dataset/` folder in the project root.

---

## 🏆 Results

### 561-Feature Models (pre-extracted features)

| Model | Test Accuracy | Notes |
|---|---|---|
| Logistic Regression | **95.52%** | Linear baseline, surprisingly strong |
| SVM (RBF, C=10) | **95.49%** | Near-best accuracy |
| LightGBM | 93.45% | 200 trees, LR=0.1 |
| XGBoost | 93.42% | 200 trees, depth=6 |
| Random Forest | 92.57% | Best feature importance |

### 45-Feature Real-Time Model (raw signals → in-browser inference)

| Model | Accuracy | Features | Where it runs |
|---|---|---|---|
| LinearSVC + Sigmoid Calibration | **86.3%** | 45 (mean/std/min/max/rms × 9 channels) | Browser JS |

### Per-Activity Accuracy (SVM, 561 features)

| Activity | Accuracy | Notes |
|---|---|---|
| 🛌 LAYING | ~100% | Gravity signal completely distinct |
| 🚶 WALKING | ~97% | Strong acceleration pattern |
| ⬇️ WALKING DOWNSTAIRS | ~96% | |
| ⬆️ WALKING UPSTAIRS | ~94% | Slightly confused with WALKING |
| 🧍 STANDING | ~90% | Hardest pair with SITTING |
| 🪑 SITTING | ~89% | Subtle gravity angle difference |

**Key findings:**
- LAYING achieves ~100% precision/recall — gravity vector is completely distinct
- SITTING vs STANDING is the hardest pair — both static, differ only in posture angle
- `tGravityAcc-mean-XYZ` features are the most discriminative across all models
- 561 hand-crafted features make linear models (LR, SVM) competitive with boosting

---

## 📁 Repository Structure

```
human-activity-recognition/
│
├── UCI HAR Dataset/                    # ⬇ Download separately (link above)
│   ├── train/
│   │   ├── X_train.txt                 # 7,352 × 561 feature matrix
│   │   ├── y_train.txt                 # Activity labels (1–6)
│   │   ├── subject_train.txt           # Subject IDs
│   │   └── Inertial Signals/           # 9 raw signal files (128 timesteps each)
│   │       ├── body_acc_x_train.txt
│   │       ├── body_acc_y_train.txt
│   │       ├── body_acc_z_train.txt
│   │       ├── body_gyro_x_train.txt
│   │       ├── body_gyro_y_train.txt
│   │       ├── body_gyro_z_train.txt
│   │       ├── total_acc_x_train.txt
│   │       ├── total_acc_y_train.txt
│   │       └── total_acc_z_train.txt
│   ├── test/                           # Same structure as train/
│   ├── features.txt                    # 561 feature names
│   ├── activity_labels.txt             # 6 class labels
│   └── features_info.txt              # Feature engineering description
│
├── har_notebook.ipynb                  # Full EDA + training + evaluation
├── train_realtime_model.py             # Train browser-compatible real-time model
├── app.py                              # Streamlit app — continuous live detection
├── models/
│   └── browser_model.json              # ✅ Committed — LinearSVC weights for JS
├── requirements.txt
├── index.html                          # GitHub Pages project page
└── README.md
```

### Generated outputs (created after running notebook + train script)

```
models/
├── svm_model.pkl                       # RBF SVM — 95.49% accuracy
├── lr_model.pkl                        # Logistic Regression — 95.52%
├── rf_model.pkl                        # Random Forest — 92.57%
├── scaler.pkl                          # StandardScaler (561 features)
├── realtime_model.pkl                  # LinearSVC on raw signals — 86.3%
├── realtime_scaler.pkl                 # StandardScaler (45 features)
├── realtime_feature_names.pkl          # Feature name list
└── browser_model.json                  # JS-ready weights (committed to repo)

Visualization outputs (saved as PNG):
├── class_distribution.png             # Bar + pie chart of activity counts
├── subject_activity_distribution.png  # Per-subject window counts (30 subjects)
├── pca_visualization.png              # 2D PCA coloured by activity
├── tsne_visualization.png             # t-SNE embedding (2,000 samples)
├── raw_signals.png                    # Body acceleration time series per activity
├── feature_means.png                  # Mean feature values per activity
├── model_comparison.png               # Horizontal bar chart of 5 model accuracies
├── confusion_matrices.png             # Confusion matrices (SVM + LR)
├── feature_importance.png             # Top 20 Random Forest feature importances
└── static_vs_dynamic.png              # Gravity scatter — static vs dynamic split
```

---

## ⚙️ Installation

### 1. Clone the repo

```bash
git clone https://github.com/rajneeshbabu/human-activity-recognition.git
cd human-activity-recognition
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm joblib matplotlib seaborn streamlit
```

### 3. Download and place the dataset

1. Download from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
2. Extract the zip
3. Place the `UCI HAR Dataset/` folder directly inside the project root

```
human-activity-recognition/
├── UCI HAR Dataset/      ← place here
├── app.py
├── har_notebook.ipynb
...
```

---

## 🚀 Running the Project — Step by Step

### Step 1 — Run the training notebook

```bash
jupyter notebook har_notebook.ipynb
```

In Jupyter: **Kernel → Restart & Run All**

What the notebook does (16 cells):
1. Imports and constants
2. Loads 561-feature train/test data
3. Class distribution — bar chart + pie chart
4. Per-subject activity window distribution
5. PCA 2D visualization
6. t-SNE 2D embedding (2,000 samples)
7. Raw inertial signal plots (body acceleration per activity)
8. Feature mean comparison across activities
9. Trains Logistic Regression (95.52%)
10. Trains SVM RBF C=10 (95.49%)
11. Trains Random Forest 200 trees (92.57%)
12. Trains XGBoost 200 trees (93.42%)
13. Trains LightGBM 200 trees (93.45%)
14. Model comparison bar chart + confusion matrices
15. Random Forest feature importance analysis
16. Static vs dynamic activity gravity scatter

Outputs saved: 10 PNG visualizations + 5 model pkl files in `models/`

### Step 2 — Train the real-time browser model

```bash
python train_realtime_model.py
```

What it does:
- Loads raw inertial signals (9 channels × 128 timesteps per window)
- Extracts 45 features: mean, std, min, max, rms per channel
- Trains LinearSVC with 5-fold sigmoid calibration
- Exports weights + scaler params to `models/browser_model.json` (32 KB)

Output:
```
Test accuracy: 86.29%
Saved → models/realtime_model.pkl | models/realtime_scaler.pkl
Exported → models/browser_model.json
```

### Step 3 — Run the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

**To open on your phone (same WiFi network):**

```bash
# macOS — find your WiFi IP
ipconfig getifaddr en0

# Linux
hostname -I

# Run accessible on the network
streamlit run app.py --server.address 0.0.0.0
```

Then open `http://<your-ip>:8501` on your phone.

---

## 📱 How to Use the Live Demo

**Online — open on your phone:**
[https://human-activity-recognition-6zwy99fqmtsp66mmprn6sn.streamlit.app/](https://human-activity-recognition-6zwy99fqmtsp66mmprn6sn.streamlit.app/)

1. Open the link on your **phone's browser**
2. Tap **▶ Start**
3. **iPhone:** tap *Allow* when prompted for motion access
4. Hold phone at your **waist**, move naturally
5. Activity prediction updates every ~0.6 seconds, continuously
6. Tap **⏹ Stop** to pause anytime

**Browser compatibility:**
- ✅ iPhone Safari — works (Safari is required on iOS for motion permission)
- ✅ Android Chrome — works automatically, no permission needed
- ❌ Chrome on iOS — blocks DeviceMotion API
- ❌ Laptop/desktop — no motion sensors

The model runs entirely in your browser — no sensor data is sent to any server.

---

## 🧠 How It Works

### Real-Time Inference Pipeline

```
Phone sensors at 50 Hz
         ↓
DeviceMotion API (JS)
         ↓
Rolling buffer — last 128 readings (2.56 s)
         ↓  every 32 new samples (~0.64 s)
Extract 45 features in JS
  (mean, std, min, max, rms × 9 channels)
         ↓
StandardScaler — stored mean/std in JSON
         ↓
LinearSVM decision function × 5 CV folds
         ↓
Sigmoid calibration → probabilities
         ↓
Average across folds → final prediction
         ↓
Update activity label + confidence bars
```

`models/browser_model.json` contains everything the JS needs: scaler mean/std arrays, SVM weight matrices (6 classes × 45 features), biases, and sigmoid calibrator A/B parameters for each of 5 cross-validation folds. Total size: 32 KB.

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `TSNE got unexpected keyword argument 'n_iter'` | scikit-learn ≥ 1.5 uses `max_iter`. Already fixed in notebook. |
| `No such file: 'models/svm_model.pkl'` | Run `har_notebook.ipynb` first (Restart & Run All). |
| `Model file not found` in app | Run `python train_realtime_model.py`, then `git add -f models/browser_model.json`. |
| `DeviceMotion not available` | Open on a real phone, not a laptop. iPhone: use Safari. |
| `Permission denied` on iOS | Tap Start again — iOS requires a direct user tap. |
| App not reachable on phone | Run with `--server.address 0.0.0.0` and use your LAN IP, not `localhost`. |
| Git push fails — lock file | Run `rm -f .git/index.lock .git/HEAD.lock` then retry. |
| `fatal: 'origin' does not appear to be a git repository` | Run `git remote add origin https://github.com/<username>/<repo>.git` |

---

## 📚 References

- Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN 2013.
- [UCI ML Repository — HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- [scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html)
- [Web DeviceMotion API — MDN](https://developer.mozilla.org/en-US/docs/Web/API/DeviceMotionEvent)

---

*© 2026 Rajneesh Babu · IISc Bengaluru*
