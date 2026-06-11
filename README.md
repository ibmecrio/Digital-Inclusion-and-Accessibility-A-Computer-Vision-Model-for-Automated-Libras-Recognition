# Libras Vision

Real-time translator for the **LIBRAS** (*Língua Brasileira de Sinais*) manual alphabet. The program captures frames from a webcam, detects the hand with MediaPipe's `HandLandmarker`, and classifies the resulting landmarks with a model trained on a local dataset. Each prediction carries a **confidence index**, and low-confidence frames are suppressed.

Only the **static** letters of the alphabet are supported — dynamic letters such as `H`, `J`, `K`, `X` and `Z` (which require motion) are out of scope.

Supported letters (20): `A B C D E F G I L M N O P Q R S T U V W`.

## How it works

1. **Hand detection** — `mediapipe` Tasks `HandLandmarker` produces 21 landmarks per detected hand, in video mode, for a single hand.
2. **Feature extraction** (`extract.py`) — landmarks are made invariant to position, handedness and scale:
   - Translated so that landmark `0` (wrist) is the origin.
   - Mirrored along the X axis when the detected hand is the left one, so the model sees every sample as if performed by a right hand.
   - Scaled by the wrist → middle-finger base distance (landmark `0` → landmark `9`), so the hand's apparent size in the frame doesn't matter.
   - The result is a 63-dimensional feature vector (21 landmarks × 3 coordinates).
3. **Classification** — by default an **SVM** (RBF kernel) from scikit-learn predicts the letter. The classifier exposes `predict_proba`, and the probability of the chosen class is used as a confidence score.
4. **Confidence gate** — a prediction is only accepted when its confidence is at least `CONFIDENCE_THRESHOLD` (default `0.7`); frames below the threshold are treated as "no confident sign" and ignored.
5. **Temporal debounce** — a repeated letter is not re-printed until at least 500 ms have passed, preventing the same sign from flooding the terminal during a sustained gesture.
6. **Display** — the frame is shown with the hand skeleton overlaid; accepted predictions are logged in the terminal as `LETTER (confidence%)`.

## Models

Five interchangeable classifiers are trained on the same 63-dimensional features. Each training script writes a `*.joblib` into `models/`:

| Script | Model | File |
| --- | --- | --- |
| `svm_model.py` | SVM (RBF, `C=10`) — **shipped by default** | `models/svm_model.joblib` |
| `mlp_model.py` | Multilayer Perceptron | `models/mlp_model.joblib` |
| `logistic_regression_model.py` | Logistic Regression | `models/logistic_regression_model.joblib` |
| `knn_model.py` | K-Nearest Neighbors (`k=21`) | `models/knn_model.joblib` |
| `random_forest_model.py` | Random Forest (300 trees) | `models/random_forest_model.joblib` |

`libras_vision.py` loads `models/svm_model.joblib`; to try another model, point that line at the corresponding `.joblib`. All models share `models/label_encoder.joblib`. The `probability=True` flag on the SVM (and `weights='uniform'` with a large `k` on the KNN) is what makes the confidence score meaningful.

A rigorous comparison of all five models — repeated stratified cross-validation, Friedman / Wilcoxon-Holm / McNemar significance tests, and confusion analysis — lives in [`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md), reproducible via the `evaluation/` suite.

## Project layout

```
.
├── dataset_estaticos/         # Image dataset grouped by letter (training/ and test/)
├── models/
│   ├── hand_landmarker.task   # MediaPipe hand landmark model
│   ├── svm_model.joblib       # Trained SVM classifier (default)
│   ├── *_model.joblib         # Other trained classifiers
│   └── label_encoder.joblib   # LabelEncoder used during training
├── extract.py                 # Landmark normalization
├── video_frame_extractor.py   # Builds dataset_estaticos/ from recorded videos
├── landmark_extractor.py      # Builds landmarks_{training,test}.csv from the dataset
├── svm_model.py               # Trains the SVM model and saves the joblib files
├── knn_model.py, mlp_model.py, ...  # Other model training scripts
├── libras_vision.py           # Main app — webcam capture, detection and prediction
├── evaluation/                # Reproducible model-comparison suite (run_all.py)
├── landmarks_training.csv     # Cached training features
├── landmarks_test.csv         # Cached test features
└── requirements.txt
```

## Requirements

- Python 3.11+ (matching the wheels in `requirements.txt`)
- A working webcam

Install the dependencies:

```bash
pip install -r requirements.txt
```

The MediaPipe model file `models/hand_landmarker.task` is required at runtime.

## Running the translator

```bash
python libras_vision.py
```

A window titled *Libras Vision - Hand Tracking* opens showing the camera feed with the hand skeleton drawn over your hand. Accepted predictions are printed in the terminal as `LETTER (confidence%)`. Press `q` or `Esc` to quit.

## The dataset

The current dataset (`dataset_estaticos/`) is built from short videos recorded by the team — five `<LETTER>_<person>.mp4` clips per letter, each of a different signer holding one static sign. `video_frame_extractor.py` samples 50 evenly spaced frames per clip and lays them out as `dataset_estaticos/{training,test}/<LETTER>/<n>.png`. The train/test split is done **by whole video** (one signer's recording per letter is held out for testing), so near-duplicate frames from the same clip never appear in both splits.

## Retraining the model

To retrain from your own data:

1. (Optional) Turn recorded videos into an image dataset:

   ```bash
   python video_frame_extractor.py
   ```

   Or place your own images as `dataset_estaticos/training/<LETTER>/*.png` (and likewise under `test/`).
2. Extract the landmarks into CSVs:

   ```bash
   python landmark_extractor.py
   ```

   This produces `landmarks_training.csv` and `landmarks_test.csv`.
3. Train a classifier and write the joblib files (e.g. the SVM):

   ```bash
   python svm_model.py
   ```

   This writes `models/svm_model.joblib` and `models/label_encoder.joblib`, which `libras_vision.py` picks up automatically.

## Evaluating the models

To reproduce the full comparative study (metrics, statistical tests, confusion matrices and `RELATORIO_TECNICO.md`):

```bash
python -m evaluation.run_all
```
