# video-intelligence-template
An example of how you use Groundlight to collect information from frames in a video and reconstruct the video with additional annotations. Also a good place to start for dealing with any timeseries problem

## Setup
1. Clone the repo and branch.
2. Ensure you have the uv package manager installed. Guide [here](https://docs.astral.sh/uv/getting-started/installation/).
3. Install the dependencies with `uv sync`.
4. Build your project on top!


## Sample Project
This repo is currently configured to do intelligence on a sample video of a Boston Dynamics robots doing sumersaults. We want to know how long the robot is upsidedown for. See the [`demo`](demo) directory for a concrete example of how to use the framework. This includes a notebook [`demo.ipynb`](demo/demo.ipynb) that shows how to use the framework from e2e.

1. Put your video (.mp4) in to the `data` folder. You can see we currently have the sample video there.
2. Use the `split_video.py` script to split the video into frames. Each frame will be saved to the `data/frames` folder. If your video is long, consider using the `--minutes` flag if you only need to process the first N minutes.

```bash
uv run framework/split_video.py data/boston_dynamics.mp4

# OR if your video is long, you can choose to only process the first N minutes
uv run framework/split_video.py data/boston_dynamics.mp4 --minutes 10
```

This will create a folder at `data/frames` with all the frames of the video:
```
data/
    frames/
        frame_0.jpg
        frame_1.jpg
        ...
        frame_1000.jpg
        ...
```

3. Update the `.env` file with your Groundlight API token. Just create the token and paste it in. Mind the instructions in the `.env` file for the type of account you should use.

4. Review the [`framework/frames.py`](framework/frames.py) file. This is a base class for handling each frame of your video and collecting metadata about it. You'll have to inherit from it for your specific use case. See the [`demo/robot_frames.py`](demo/robot_frames.py) file for a concrete example. At minimum, you'll have to implement the `process_frame` method, which defines the metadata you want to collect about the frame. Optionally, you can also implement the `update_frame` method, which will update the metadata by re-querying the Groundlight API for an updated result (e.g. human labeling or reprediction). Refrain from putting stateful logic in one's frame class - e.g. logic that depends on the results of previous or future frames, as this will prevent you from taking advantage of the multithreading capabilities of the framework. Put stateful logic in one's analysis function instead (described below).

5. Warm up your detectors with the `warm_up_frame_class` method. This randomly samples from your frames and sends them to GL to warm up the relevant detectors.

```python
from robot_frames import RobotFrame
from framework.utils import warm_up_frame_class

warm_up_frame_class(frame_class=RobotFrame, proportion=0.1)

```

6. Process your frames using the prefetcher, which sends the remaining frames to GL to process:
```python
from tqdm.auto import tqdm
from robot_frames import RobotFrame

from framework.prefetcher import FramePrefetcher
from framework.utils import (
    get_first_frame_index,
    get_last_frame_index,
)

start_index = get_first_frame_index()
end_index = get_last_frame_index()
indicies = list(range(start_index, end_index + 1))

prefetcher = FramePrefetcher(
    frame_class=RobotFrame,
    indicies=indicies,
    action="process",
)
for index in tqdm(indicies, desc="Processing frames"):
    frame = prefetcher.get_frame(index)
```

7. Update your frames with the latest answers from GL (e.g. reprediction or human labeling):
```python
from tqdm.auto import tqdm
from robot_frames import RobotFrame
from framework.prefetcher import FramePrefetcher


prefetcher = FramePrefetcher(
    frame_class=RobotFrame,
    indicies=indicies,
    action="update",
)
for index in tqdm(indicies, desc="Updating frames"):
    frame = prefetcher.get_frame(index)

```

8. Check what proportion of your frames have answers:
```python
from framework.utils import proportion_with_answer
from robot_frames import RobotFrame

proportion = proportion_with_answer(
    frame_class=RobotFrame,
    indices=indicies,
    has_answer_function=RobotFrame.has_answer
)
print(f"Proportion of frames with answers: {proportion}")
```

9. Define your analysis function. This is doing your "application logic", e.g. tracking, counting, aggregating, etc. See the [`demo/robot_analysis_function.py`](demo/robot_analysis_function.py) file for a concrete example. Put your stateful logic here. Your analysis function can be a class, allowing you to maintain arbitrarily complex state across frames.


10. Run your analysis function on your frames, producing a new video and whatever other outputs you want.
```python
from robot_analysis_function import RobotUpsideDownAnalysis
from robot_frames import RobotFrame
from framework.process_frames import process_frames

analysis_class = RobotUpsideDownAnalysis()


process_frames(
    run_name="robot_upside_down_detection",
    indices=indicies,
    output_path="output.mp4",
    analysis_function=analysis_class.analyze_frame,
    frame_class=RobotFrame,
    input_video_path="../data/boston_dynamics.mp4",
)
```