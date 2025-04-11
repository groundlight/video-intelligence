import json
import os
from abc import abstractmethod

import cv2
import numpy as np
from dotenv import load_dotenv
from groundlight import Groundlight
from pydantic import BaseModel, ConfigDict

from .constants import FRAMES_METADATA_PATH, FRAMES_PATH

# loads the Groundlight API token from the .env file
load_dotenv()


class ProcessedFrame(BaseModel):
    """
    A frame's process_frame must return a ProcessedFrame object. This object allows arbitrary metadata to be stored about the frame in additon to some required fields.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    index: int
    image: np.ndarray


class Frame:
    """
    This base class represents a frame of a video. It includes:
    1. The frame loaded as an image in BGR format
    2. Metadata about the frame loaded/saved to disk
    3. The Groundlight SDK object stored in self._gl. There is one instance of this per class.
    """

    _gl = None

    def __init__(self, index, initial_metadata=None):
        """
        index: the index of the frame in the video. Must correspond to frame_<index>.jpg in the frames directory.

        initial_metadata: a dictionary of metadata to initialize the frame with. This will get updated as you process the frame.
        By default, the metadata will be an empty dictionary.
        """

        if self._gl is None:
            print(
                "No groundlight client initialized before instantiating frame class. Initializing to use prod endpoint. Use initialize_gl() to initialize a client for a different endpoint."
            )
            self.initialize_gl()

        self.index = index

        self.frame_path = f"{FRAMES_PATH}/frame_{index}.jpg"
        if not os.path.exists(self.frame_path):
            raise FileNotFoundError(f"Frame {self.frame_path} does not exist")

        # where we will store the image. We will load it lazily
        self._image = None

        # we will store metadata about the frame
        self.metadata_path = f"{FRAMES_METADATA_PATH}/frame_{index}.json"

        # check if metadata already exists, if not create it and save it to the metadata path. Otherwise, load it
        self.initial_metadata = initial_metadata or {}
        if not isinstance(self.initial_metadata, dict):
            raise ValueError("initial_metadata must be a dictionary")

        self.load_metadata()

    @classmethod
    def initialize_gl(cls, endpoint=None):
        """
        Initializes the Groundlight SDK object.
        """
        cls._gl = Groundlight(endpoint)

    @property
    def image(self):
        """
        Lazily load the image in BGR format
        """
        if self._image is None:
            self._image = cv2.imread(self.frame_path)  # type: ignore
        return self._image

    def save_metadata(self) -> None:
        """
        Saves self.metadata to self.metadata_path
        """
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    def load_metadata(self) -> dict:
        """
        Loads self.metadata from self.metadata_path if it exists. Otherwise it saves self.initial_metadata to self.metadata_path.
        Either way stores the metadata in self.metadata.
        """
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)

            # Handle the case where the user updates the initial metadata with new keys after metadata has already been saved
            # Only add keys that don't already exist in the loaded metadata.
            for key, value in self.initial_metadata.items():
                if key not in existing_metadata:
                    existing_metadata[key] = value
            self.metadata = existing_metadata
            self.save_metadata()

        else:
            # If metadata doesn't exist, use the initial metadata
            self.metadata = self.initial_metadata
            self.save_metadata()

    @abstractmethod
    def process_frame(self) -> ProcessedFrame:
        """
        Does metadata collection about the frame and saves it to disk. Must return a ProcessedFrame object.
        This operation should NOT be stateful, as in it should not rely on any previous or future frame indicies.
        Stateful operationsshould take place in one's analysis function.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def update_frame(self):
        """
        Updates existing metadata by re-querying the Groundlight API for an updated result (e.g. human labeling or reprediction).
        By default, this is not implemented and can be overridden by subclasses.
        This operation should NOT be stateful, as in it should not rely on any previous or future frame indicies.
        Stateful operations should take place in one's analysis function.
        """
        raise NotImplementedError("Subclasses must implement this method")
