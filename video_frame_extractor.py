import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Videos are named "<LETTER>_<person>.mp4" (e.g. "A_Patrick.mp4"), 5 per letter.
# Each video is ~10s of an interpreter holding one static sign. We sample a fixed
# number of evenly spaced frames per video and lay them out exactly like the
# existing "dataset" folder: <output>/{training,test}/<LETTER>/<n>.png
#
# The train/test split is done by whole video (one person's recording is held out
# for test) instead of by frame, so frames from the same recording never appear in
# both sets, which would leak information and inflate the test score.

DEFAULT_INPUT_DIR = Path.home() / "OneDrive" / "Pictures" / "Camera Roll" / "Estaticos"
DEFAULT_OUTPUT_DIR = Path("dataset_estaticos")
FRAMES_PER_VIDEO = 50
TEST_VIDEOS_PER_LETTER = 1  # held out for the test set; the rest go to training
MAX_SIZE = 256  # longest side of the saved frame, in pixels (aspect ratio kept)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def resize_keep_aspect(frame, max_size):
    """Scale `frame` so its longest side is `max_size`, preserving aspect ratio.

    The source videos are 16:9; squishing them into a square would warp the
    hand and corrupt the landmark features extracted downstream, so the aspect
    ratio is preserved. Frames already within the limit are returned unchanged.
    """
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return frame
    scale = max_size / longest
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_frames(video_path, num_frames):
    """Return up to `num_frames` evenly spaced BGR frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: could not open {video_path.name}, skipping.")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print(f"  WARNING: {video_path.name} reports 0 frames, skipping.")
        cap.release()
        return []

    # Evenly spaced indices across the whole clip (inclusive of first and last).
    indices = np.linspace(0, total - 1, num=num_frames, dtype=int)

    frames = []
    for target in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        success, frame = cap.read()
        if success:
            frames.append(frame)
        else:
            print(f"  WARNING: failed to read frame {target} of {video_path.name}.")

    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Turn the Estaticos videos into an image dataset of evenly "
                    "spaced frames, split into training/test by video."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR,
                        help=f"Folder with the source videos (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Destination dataset folder (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--frames", type=int, default=FRAMES_PER_VIDEO,
                        help=f"Frames to extract per video (default: {FRAMES_PER_VIDEO})")
    parser.add_argument("--test-videos", type=int, default=TEST_VIDEOS_PER_LETTER,
                        help=f"Videos per letter held out for test (default: {TEST_VIDEOS_PER_LETTER})")
    parser.add_argument("--max-size", type=int, default=MAX_SIZE,
                        help=f"Longest side of saved frames in px, aspect kept (default: {MAX_SIZE})")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    # Group videos by letter (the part before the first underscore), uppercased.
    videos_by_letter = defaultdict(list)
    for video_path in sorted(input_dir.iterdir()):
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        letter = video_path.stem.split("_")[0].upper()
        videos_by_letter[letter].append(video_path)

    if not videos_by_letter:
        raise SystemExit(f"No video files found in {input_dir}")

    # Start from a clean output folder so reruns don't leave stale frames behind.
    if output_dir.exists():
        print(f"Removing existing {output_dir} ...")
        shutil.rmtree(output_dir)

    total_train = 0
    total_test = 0

    for letter in sorted(videos_by_letter):
        videos = sorted(videos_by_letter[letter])  # deterministic order
        # Hold out the last N videos for test, the rest for training.
        n_test = min(args.test_videos, len(videos))
        train_videos = videos[:len(videos) - n_test]
        test_videos = videos[len(videos) - n_test:]

        print(f"\nLetter {letter}: {len(train_videos)} train video(s), "
              f"{len(test_videos)} test video(s)")

        for split, split_videos in (("training", train_videos), ("test", test_videos)):
            letter_dir = output_dir / split / letter
            letter_dir.mkdir(parents=True, exist_ok=True)

            counter = 1  # frames numbered sequentially per letter/split folder
            for video_path in split_videos:
                frames = extract_frames(video_path, args.frames)
                for frame in frames:
                    frame = resize_keep_aspect(frame, args.max_size)
                    cv2.imwrite(str(letter_dir / f"{counter}.png"), frame)
                    counter += 1
                print(f"  [{split}] {video_path.name}: {len(frames)} frames")

            saved = counter - 1
            if split == "training":
                total_train += saved
            else:
                total_test += saved

    print(f"\nDone. {total_train} training frames, {total_test} test frames "
          f"written to {output_dir}/")


if __name__ == "__main__":
    main()
