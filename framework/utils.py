import random
from math import ceil
from typing import Callable, Optional, Type

from tqdm import tqdm

from .constants import FRAMES_PATH
from .frames import Frame
from .prefetcher import FramePrefetcher


def _get_frame_indices() -> list[int]:
    return [
        int(frame.stem.split("_", 1)[1]) for frame in FRAMES_PATH.glob("frame_*.jpg")
    ]


def get_first_frame_index() -> Optional[int]:
    """
    Returns the first frame index in the frames directory.
    """

    indices = _get_frame_indices()
    if not indices:
        raise ValueError("No frames found in the frames directory")
    return min(indices)


def get_last_frame_index() -> Optional[int]:
    """
    Returns the last frame index in the frames directory.
    """

    indices = _get_frame_indices()
    if not indices:
        raise ValueError("No frames found in the frames directory")
    return max(indices)


def proportion_with_answer(
    frame_class: Type[Frame],
    indices: list[int],
    has_answer_function: Callable[[Frame], bool],
) -> float:
    """
    Returns the proportion of frames in the given indices that have an answer.

    frame_class: the class of the frames to check

    indices: the indices of the frames to check

    has_answer_function: a function that takes a frame and returns True if it has an answer, False otherwise
    """

    if len(indices) == 0:
        raise ValueError("No indices provided")

    has_answer_results = []
    for index in indices:
        frame = frame_class(index)
        has_answer_results.append(has_answer_function(frame))

    return sum(has_answer_results) / len(has_answer_results)


def warm_up_frame_class(
    frame_class: Type[Frame],
    proportion: float | None = None,
    indicies: list[int] | None = None,
) -> None:
    """
    Warms up the given frame class by processing a random subset of frames.

    If proportion is provided, a random subset of all of the frames will be processed.
    If indicies is provided, all frames in indicies will be processed (useful for separating training from testing data,
    e.g. train on the first N minutes, test on remaining M minutes).

    Parameters:
      frame_class: The class derived from Frame to be warmed up.
      proportion: A float in the range (0.0, 1.0] representing the proportion of frames to process.
                Must be > 0 and <= 1.
      indicies: A list of frame indices to process.
    """
    if proportion is None and indicies is None:
        raise ValueError("Either proportion or indicies must be provided")
    elif proportion is not None and indicies is not None:
        raise ValueError("Only one of proportion or indicies can be provided")

    # if indicies is not provided, choose a random subset of all frames
    if indicies is None:
        start = get_first_frame_index()
        end = get_last_frame_index()
        indicies = list(range(start, end + 1))
        random.shuffle(indicies)
        if not (0 < proportion <= 1):
            raise ValueError(f"Proportion must be in (0, 1]. Received: {proportion}")
        indicies = indicies[: ceil(len(indicies) * proportion)]
    # otherwise, use the provided indicies
    else:
        indicies = list(indicies)
        if len(indicies) == 0:
            raise ValueError("Indicies must contain at least value")

    print(f"Warming up with {len(indicies)} frames")

    prefetcher = FramePrefetcher(
        frame_class=frame_class,
        indicies=indicies,
        action="process",
    )

    for index in tqdm(indicies, desc="Warming up frame class"):
        prefetcher.get_frame(index)
