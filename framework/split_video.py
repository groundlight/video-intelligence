import argparse
import os

import cv2
from tqdm import tqdm

from constants import FRAMES_PATH

# splits the videos into frames and saves them to the frames_path directory


def split_video(video_path, frames_path, minutes=None):
    # Ensure the frames_path directory exists
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    # verify the frames path is empty
    if os.listdir(frames_path):
        raise ValueError(
            f"Frames path {frames_path} is not empty. Please clear it before running the script."
        )

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get fps and calculate total frames to process
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if minutes:
        frames_to_process = min(int(minutes * 60 * fps), total_frames)
    else:
        frames_to_process = total_frames

    print(
        f"Processing {frames_to_process} frames out of {total_frames} total frames at {fps} fps"
    )

    # Save each frame to the frames_path directory
    for frame_number in tqdm(range(int(frames_to_process)), desc="Splitting video"):
        frame_path = f"{frames_path}/frame_{frame_number}.jpg"
        if os.path.exists(frame_path):
            continue

        ret, frame = cap.read()
        if ret:
            success = cv2.imwrite(frame_path, frame)
            if not success:
                print(f"Failed to save frame {frame_number}")
        else:
            print(f"Failed to read frame {frame_number}")

    # Release the video capture object
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split video into frames")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument(
        "--minutes", type=float, help="Number of minutes to process (optional)"
    )

    args = parser.parse_args()

    if args.minutes:
        print(
            f"Splitting first {args.minutes} minutes of video {args.video_path} into frames at {FRAMES_PATH}"
        )
    else:
        print(f"Splitting video {args.video_path} into frames at {FRAMES_PATH}")

    split_video(args.video_path, FRAMES_PATH, args.minutes)
