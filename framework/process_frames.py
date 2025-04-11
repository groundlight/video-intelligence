from pathlib import Path
from typing import Callable, Type

import cv2
from cv2 import CAP_PROP_FPS
from tqdm.auto import tqdm

from .frames import Frame, ProcessedFrame
from .prefetcher import FramePrefetcher


def process_frames(
    *,
    run_name: str,
    indices: list[int],
    output_path: str | None,
    analysis_function: Callable[[ProcessedFrame], ProcessedFrame],
    frame_class: Type[Frame],
    input_video_path: str,
) -> None:
    """
    Process a set of frames in the order specified by the indices. If output_path is provided, a video will be produced.
    If analysis_function is provided, it will be called for each frame. The output of the analysis function will be the frame that is written to the video.

    run_name: the name of the run. Passed in the FrameInformation object to the analysis function to be used as a unique identifier for your analysis.
    indices: the indices of the frames to process.
    output_path: Where to save the output video. If None, no video will be produced. This speeds up processing if you don't need the video.
    analysis_function: the function to call for each frame. This consumes a ProcessedFrame object and must return a ProcessedFrame object. The output image is used to generate the video, so if you draw on it, these annotations will be included in the video.
    frame_class: the class of the frames to process.
    """

    pool_size = min(len(indices), 32)

    prefetcher = FramePrefetcher(
        frame_class=frame_class,
        indicies=indices,
        buffer_size=pool_size,
        num_workers=pool_size,
    )

    # verify the input video exists
    if not Path(input_video_path).exists():
        raise FileNotFoundError(f"Input video {input_video_path} does not exist")

    # get the input video FPS so we can write the output video at the same FPS
    fps = cv2.VideoCapture(input_video_path).get(CAP_PROP_FPS)

    # verify non zero FPS. If zero, the video will not write.
    if fps <= 0.0:
        raise ValueError(
            f"Input video {input_video_path} has a FPS of {fps}. Must be positive"
        )

    if output_path is not None:
        # Get the dimensions of the video
        first_frame = frame_class(indices[0]).image
        frame_height, frame_width = first_frame.shape[:2]

        output_path = Path(output_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (frame_width, frame_height)
        )
    else:
        print("No output path for video provided, will not produce video")

    for index in tqdm(indices):
        # get the ProcesseFrame from the prefetcher
        processed_frame = prefetcher.get_frame(index)["result"]

        # Call the analysis function on the processed frame, if provided
        # The analysis function must return a ProcessedFrame object, it can modify it as needed, or it can return it unchanged
        if analysis_function is not None:
            processed_frame = analysis_function(processed_frame)

        if not isinstance(processed_frame, ProcessedFrame):
            raise ValueError(
                "We must have a ProcessedFrame object to write to the video"
            )

        # write the processed frame to the video
        if output_path is not None:
            video_writer.write(processed_frame.image)
