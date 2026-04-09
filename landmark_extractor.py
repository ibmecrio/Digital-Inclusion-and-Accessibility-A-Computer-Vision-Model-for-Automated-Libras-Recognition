import pandas as pd
import mediapipe as mp
import extract
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

axes = ['x', 'y', 'z']
NUM_LANDMARKS = 21
counter = 0

dataset_path = Path("dataset/training")

landmarks_dict = dict()
labels = list()

for idx in range(NUM_LANDMARKS):
    landmarks_dict[f'x{idx}'] = list()
    landmarks_dict[f'y{idx}'] = list()
    landmarks_dict[f'z{idx}'] = list()


base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

for letter_dir in dataset_path.iterdir():
    if not letter_dir.is_dir():
        continue
    
    current_letter = letter_dir.name
    
    for image_path in letter_dir.glob("*.png"):
        image = mp.Image.create_from_file(str(image_path))
        detection_result = detector.detect(image)

        treated_landmarks = extract.extract_relative_coords(detection_result)

        if treated_landmarks.size != 0:
            for idx in range(NUM_LANDMARKS*3):
                axis = axes[idx % 3]
                landmark_num = idx // 3
                landmarks_dict[f'{axis}{landmark_num}'].append(treated_landmarks[idx].item())

            labels.append(current_letter)
            print(counter)
            counter += 1

landmarks_dict['label'] = labels

landmarks_df = pd.DataFrame(landmarks_dict)
landmarks_df.to_csv('landmarks_training.csv')