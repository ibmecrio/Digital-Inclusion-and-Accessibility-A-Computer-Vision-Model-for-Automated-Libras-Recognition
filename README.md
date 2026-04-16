# Libras Vision

Real-time translator for the **LIBRAS** (*Língua Brasileira de Sinais*) manual alphabet. The program captures frames from a webcam, detects the hand with MediaPipe's `HandLandmarker`, and classifies the resulting landmarks with a KNN model trained on a local dataset.

Only the **static** letters of the alphabet are supported — dynamic letters such as `H`, `J`, `K`, `X` and `Z` (which require motion) are out of scope.

Supported letters: `A B C D E F G I L M N O P Q R S T U V W Y`.

## How it works

1. **Hand detection** — `mediapipe` Tasks `HandLandmarker` produces 21 landmarks per detected hand, in video mode, for a single hand.
2. **Feature extraction** (`extract.py`) — landmarks are made invariant to position, handedness and scale:
   - Translated so that landmark `0` (wrist) is the origin.
   - Mirrored along the X axis when the detected hand is the left one, so the model sees every sample as if performed by a right hand.
   - Scaled by the wrist → middle-finger base distance (landmark `0` → landmark `9`), so the hand's apparent size in the frame doesn't matter.
3. **Classification** — a `KNeighborsClassifier` (`k = 5`) from scikit-learn predicts the letter from the 63-dimensional feature vector (21 landmarks × 3 coordinates).
4. **Display** — the frame is shown with the hand skeleton overlaid; predictions are logged in the terminal.

## Project layout

```
.
├── dataset/                   # Images grouped by letter (training/ and test/)
├── models/
│   ├── hand_landmarker.task   # MediaPipe hand landmark model
│   ├── knn_model.joblib       # Trained KNN classifier
│   └── label_encoder.joblib   # LabelEncoder used during training
├── extract.py                 # Landmark normalization
├── landmark_extractor.py      # Builds landmarks_training.csv from dataset/training
├── knn_model.py               # Trains the KNN model and saves the joblib files
├── libras_vision.py           # Main app — webcam capture, detection and prediction
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

A window titled *Libras Vision - Hand Tracking* opens showing the camera feed with the hand skeleton drawn over your hand. Predictions are printed in the terminal. Press `q` or `Esc` to quit.

## Retraining the model

To retrain from your own images:

1. Place one subdirectory per letter inside `dataset/training/` (e.g. `dataset/training/A/*.png`).
2. Extract the landmarks into a CSV:

   ```bash
   python landmark_extractor.py
   ```

   This produces `landmarks_training.csv`.
3. Train the classifier and write the new joblib files:

   ```bash
   python knn_model.py
   ```

   Move the generated `knn_model.joblib` and `label_encoder.joblib` into `models/` so `libras_vision.py` picks them up.
