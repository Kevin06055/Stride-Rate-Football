import supervision as sv
from tqdm import tqdm

def read_video(source):
    frame_generator = sv.get_video_frames_generator(source)
    frames = []
    for frame in frame_generator:
        frames.append(frame)

    return frames
def save_video(output_frames, SOURCE_VIDEO_PATH, output_video_path, frames_per_second=24):
    """
    Save video frames to a new video file.

    Args:
        output_frames (list): List of frames to save as a video.
        SOURCE_VIDEO_PATH (str): Path to the source video for video information.
        output_video_path (str): Path where the output video will be saved.
        frames_per_second (int): Frames per second for the output video. Default is 24.
    """
    # Use from_video_path to get video info
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    # Adjust fps if necessary
    video_info.fps = frames_per_second

    # Create a VideoSink with the video information
    video_sink = sv.VideoSink(output_video_path, video_info)

    # Write frames to the video using tqdm for progress
    with video_sink:
        for frame in tqdm(output_frames, desc="Saving video", unit="frame"):
            video_sink.write_frame(frame)