from groundlight import Detector, ImageQuery, Label

from framework.frames import Frame, ProcessedFrame


class RobotFrame(Frame):
    """
    Frame class for Boston Dynamics robot video.
    """

    # all instances of this class will share the same robot detector
    _robot_detector = None

    def __init__(self, index):
        # we store the iq_id (so we can query for it later) and the current answer for the frame.
        initial_metadata = {
            "iq_id": None,  # the iq id used for this frame (str)
            "is_upside_down": None,  # the current best answer for the frame (bool)
        }

        super().__init__(index=index, initial_metadata=initial_metadata)

    @property
    def robot_detector(self) -> Detector:
        """
        Lazily initialize and return the robot detector - shared across all instances of this class.
        """
        cls = type(self)
        if cls._robot_detector is None:
            cls._robot_detector = self._gl.get_or_create_detector(
                name="robot_detector",
                query="This is a robot doing a sumersault. Is the robot currently upside down (feet above head)?",
                confidence_threshold=0.9,
            )
        return cls._robot_detector

    def _update_metadata(self, iq: ImageQuery):
        """
        Given an iq, updates the metadata appropiately and saves it
        """

        self.metadata["iq_id"] = iq.id

        # we store the result as long as we get a confident and clear answer.
        confidence = iq.result.confidence
        if (
            confidence >= self.robot_detector.confidence_threshold
            and iq.result.label != Label.UNCLEAR
        ):
            self.metadata["is_upside_down"] = iq.result.label == Label.YES

        self.save_metadata()

    def process_frame(self) -> ProcessedFrame:
        """
        Process the frame to collect metadata by querying the robot detector.
        Returns a ProcessedFrame object with the metadata.
        """
        # Processing this frame is only necessary if it doesn't already have an iq_id. Use update_frame() to query for an updated answer."
        if self.metadata["iq_id"] is None:
            iq = self._gl.ask_ml(image=self.image, detector=self.robot_detector)
            self._update_metadata(iq)

        return ProcessedFrame(
            index=self.index, image=self.image, metadata=self.metadata
        )

    def update_frame(self):
        if self.metadata["iq_id"] is None:
            raise ValueError(
                f"Frame {self.index} has no iq_id. Use process_frame() to query for an answer first before trying to update."
            )

        # we only need to query for an updated answer if we don't already have one
        if self.metadata["is_upside_down"] is None:
            iq = self._gl.get_image_query(self.metadata["iq_id"])
            self._update_metadata(iq)

    @staticmethod
    def has_answer(frame: "RobotFrame") -> bool:
        """
        Returns true if the given RobotFrame has an answer.
        """
        return frame.metadata["is_upside_down"] is not None
