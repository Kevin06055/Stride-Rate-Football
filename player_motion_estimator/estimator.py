import cv2
import sys
sys.path.append('../')
from utility import measure_distance, get_foot_position

class gait_metrics_estimator():
    """
    Class for estimating gait metrics like speed, distance, and stride rate for tracked objects.
    """
    def __init__(self, frame_rate=24):
        """
        Initialize the estimator with frame rate and sliding window size.

        :param frame_rate: Frames per second for video processing.
        """
        self.frame_window = 5  # Sliding window for speed estimation
        self.frame_rate = frame_rate

        self.tracks = {}
        self.metrics = {}

    def add_metrics(self, tracks):
        """
        Add estimated metrics (speed, distance, stride rate) to the track information.
        
        :param tracks: Dictionary containing the tracked objects.
        """
        total_distance = {}

        for object_type, object_tracks in tracks.items():
            if object_type not in ["players", "goalkeepers"]:
                continue

            for tracker_id, track_data in object_tracks.items():

                if tracker_id not in total_distance:
                    total_distance[tracker_id] = 0

                for frame_num in sorted(track_data.keys()):
                    # Ensure that frame_num + frame_window exists in track_data
                    if (frame_num + self.frame_window) not in track_data:
                        continue  # Skip if the window extends beyond available frames

                    start_position = track_data.get(frame_num, {}).get('position_transformed')
                    end_position = track_data.get(frame_num + self.frame_window, {}).get('position_transformed')

                    if not start_position or not end_position:
                        continue

                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = self.frame_window / self.frame_rate
                    speed_mps = distance_covered / time_elapsed if time_elapsed > 0 else 0
                    speed_kmph = speed_mps * 3.6

                    total_distance[tracker_id] += distance_covered

                    average_stride_length = 1.5  # Placeholder, you can adjust it as needed
                    stride_rate = (speed_mps / average_stride_length) * 60 if average_stride_length > 0 else 0

                    # Add metrics to the track data for the current window
                    for i in range(frame_num, frame_num + self.frame_window):
                        if i not in track_data:
                            continue

                        track_data[i].setdefault('metrics', {})
                        track_data[i]['metrics']['speed_kmph'] = speed_kmph/1000
                        track_data[i]['metrics']['distance'] = total_distance[tracker_id]
                        track_data[i]['metrics']['stride_rate'] = stride_rate

        self.tracks = tracks
        self.metrics = total_distance

    def get_metrics(self):
        """
        Retrieve computed metrics.

        :return: Dictionary of computed metrics for tracked objects.
        """
        return self.metrics

    def get_tracks(self):
        """
        Retrieve updated tracks with metrics.

        :return: Dictionary of tracks with added metrics.
        """
        return self.tracks
