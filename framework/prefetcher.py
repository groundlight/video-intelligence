import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Type

from .frames import Frame


class FramePrefetcher:
    def __init__(
        self,
        frame_class: Type[Frame],
        # start_frame_index: int,
        indicies: list[int],
        buffer_size: int = 8,
        num_workers: int = 8,
        action: str = "process",
    ):
        """
        Keeps a buffer of frames in memory to speed up processing.

        frame_class: the class to use to load frames. Must inherit from Frame class.

        indicies: the indicies of the frames to load, in order

        buffer_size: the number of frames to prefetch

        num_workers: the number of threads to use to load frames, generally set to the number of cores
        on your machine.

        action: the action to call on the frame class. Must be one of "process" or "update".
        """

        self.frame_class = frame_class

        self.buffer_size = buffer_size

        # the index of indicies list we expect to be requested next
        self.current_index = 0

        # the indicies of frames to load, in order. This list does not necessarily need to be in order, e.g. [2,1,0] is acceptable.
        self.indicies = indicies

        # maps frame indices to the result of the action on the frame
        self.buffer: Dict[int, Any] = {}

        # Set the function reference based on action.
        if action == "process":
            self.action_fn = lambda frame: frame.process_frame()
        elif action == "update":
            self.action_fn = lambda frame: frame.update_frame()
        else:
            raise ValueError(
                f"Action must be one of 'process' or 'update', got {action}"
            )

        # Set to track frame indicies currently being acted on (processed or updated)
        self.in_progress = set()

        # Thread pool and lock
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.lock = threading.Lock()

        # Start prefetching
        self._schedule_prefetch()

    def _schedule_prefetch(self):
        """Schedule prefetch tasks for upcoming frames"""
        with self.lock:
            # Calculate which frames we need
            current_buffer_indices = set(self.buffer.keys())
            needed_indices = set(
                self.indicies[
                    self.current_index : self.current_index + self.buffer_size
                ]
            )
            # Don't fetch frame indicies that are either in buffer or already being fetched
            indices_to_fetch = (
                needed_indices - current_buffer_indices - self.in_progress
            )

            for idx in indices_to_fetch:
                self.in_progress.add(idx)
                future = self.executor.submit(self._get_frame, idx)
                future.add_done_callback(self._store_frame_callback)

    def _store_frame_callback(self, future):
        """Callback to store completed frame in buffer"""
        try:
            result = future.result()
            with self.lock:
                self.buffer[result["index"]] = result
                # Remove from in_progress set once complete
                self.in_progress.remove(result["index"])
        except Exception as e:
            print(f"Error fetching frame: {e}")
            raise e

    def _get_frame(self, index: int) -> Frame:
        """
        Called by worker to load a frame
        """
        frame = self.frame_class(index)
        result = self.action_fn(frame)
        return {"index": index, "result": result}

    def get_frame(self, index: int):
        """
        Called by consumer loop to get a frame.

        Returns a dictionary with two keys:
        - "index": the index of the frame
        - "result": the result of what calling the action on the frame returns
        """
        if index != self.indicies[self.current_index]:
            raise ValueError(
                f"Index requested {index} is out of order. This class expects frames to be consumed in the order specified by indicies passed to the constructor. It expected {self.indicies[self.current_index]} to be requested next."
            )

        # Wait for the requested frame
        while self.indicies[self.current_index] not in self.buffer:
            time.sleep(0.01)

        with self.lock:
            data = self.buffer.pop(self.indicies[self.current_index])
            self.current_index += 1

        # Schedule next batch of frames
        self._schedule_prefetch()

        return data
