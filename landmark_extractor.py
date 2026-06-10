import argparse
import pandas as pd
import mediapipe as mp
import extract
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

axes = ['x', 'y', 'z']
NUM_LANDMARKS = 21
NUM_COORDS = NUM_LANDMARKS * 3  # 63 landmark coordinates

# Default source dataset and the CSV each split is written to. The training
# scripts read these two filenames, so writing here makes the new data train-ready.
DEFAULT_DATASET_DIR = Path("dataset_estaticos")
SPLITS = {
    "training": "landmarks_training.csv",
    "test": "landmarks_test.csv",
}

base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)


def build_split(split_dir):
    """Run hand detection over every image in a split and return a DataFrame
    of the 63 relative landmark coordinates plus a 'label' column."""
    landmarks_dict = dict()
    for idx in range(NUM_LANDMARKS):
        landmarks_dict[f'x{idx}'] = list()
        landmarks_dict[f'y{idx}'] = list()
        landmarks_dict[f'z{idx}'] = list()
    labels = list()
    counter = 0

    for letter_dir in sorted(split_dir.iterdir()):
        if not letter_dir.is_dir():
            continue

        current_letter = letter_dir.name

        for image_path in sorted(letter_dir.glob("*.png")):
            image = mp.Image.create_from_file(str(image_path))
            detection_result = detector.detect(image)

            # Static images have no previous frame, so there is no wrist motion:
            # pass last_wrist_position=None (velocity becomes 0) and keep only the
            # 63 coordinate features, matching what the models consume.
            treated_landmarks, _ = extract.extract_relative_coords(
                detection_result, 0, None
            )

            if treated_landmarks.size != 0:
                for idx in range(NUM_COORDS):
                    axis = axes[idx % 3]
                    landmark_num = idx // 3
                    landmarks_dict[f'{axis}{landmark_num}'].append(treated_landmarks[idx].item())

                labels.append(current_letter)
                counter += 1
                print(f"  {split_dir.name}: {counter} samples", end="\r")

    print()
    landmarks_dict['label'] = labels
    return pd.DataFrame(landmarks_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from an image dataset and "
                    "write one CSV per split (training/test)."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR,
                        help=f"Dataset folder with training/ and test/ subdirs "
                             f"(default: {DEFAULT_DATASET_DIR})")
    args = parser.parse_args()

    for split, csv_name in SPLITS.items():
        split_dir = args.dataset / split
        if not split_dir.is_dir():
            print(f"Skipping '{split}': {split_dir} not found.")
            continue

        print(f"Processing '{split}' from {split_dir} ...")
        df = build_split(split_dir)
        df.to_csv(csv_name)
        print(f"Wrote {csv_name}: {len(df)} rows, {df['label'].nunique()} letters.")


if __name__ == "__main__":
    main()
