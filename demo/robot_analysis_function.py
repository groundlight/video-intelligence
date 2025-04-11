# This is my analysis function for the robot video.
# It will do the following:
# 1. Statefully track how many frames the robot is upside down
# 2. Draw the state on each frame

# Notice that this class can be arbitrarily complex and/or stateful. You could keep track of the previous frames, or whatever else you need.

import cv2
import numpy as np

from framework.frames import ProcessedFrame


class RobotUpsideDownAnalysis:
    def __init__(self):
        self.robot_upside_down_frame_count = 0

    def analyze_frame(self, frame: ProcessedFrame) -> ProcessedFrame:
        metadata = frame.metadata
        is_upside_down = metadata["is_upside_down"]

        # update the state if appropriate
        if is_upside_down:
            self.robot_upside_down_frame_count += 1

        # draw the current state onto the frame
        frame.image = self.draw_state(frame.image)

        # return the modified frame so it can be written to the video
        return frame

    def draw_state(self, image) -> np.ndarray:
        text = f"Number of frames upside down: {self.robot_upside_down_frame_count}"

        # Get image dimensions and calculate text position
        height, width = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2

        # Get text size to position it properly
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        text_x = 10
        text_y = height - 20

        # Draw text with black background for better visibility
        cv2.putText(
            image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1
        )
        cv2.putText(
            image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness
        )

        return image
