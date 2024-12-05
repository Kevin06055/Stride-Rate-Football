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

        self.tracks={}
        self.metrics={}

    def add_metrics(self, tracks, detections_dict):
        """
        Add estimated metrics (speed, distance, stride rate) to the track information.
        
        :param tracks: Dictionary containing the tracked objects.
        :param detections_dict: Dictionary containing detection data for objects.
        """
        total_distance = {}

        for object_type, object_tracks in tracks.items():
            if object_type in ["ball", "referee"]:
                continue

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for tracker_id in object_tracks[frame_num].keys():
                    if tracker_id in detections_dict['tracker_id']:
                        detections_idx = detections_dict['tracker_id'].index(tracker_id)
                        
                        confidence = detections_dict['confidence'][detections_idx]
                        class_id = detections_dict['class_id'][detections_idx]
                        
                        if confidence < 0.6:
                            continue

                        start_position = object_tracks[frame_num][tracker_id].get('position_transformed', None)
                        end_position = object_tracks[last_frame][tracker_id].get('position_transformed', None)

                        if start_position is None or end_position is None:
                            continue

                        # Metrics Calculation
                        distance_covered = measure_distance(start_position, end_position)
                        time_elapsed = (last_frame - frame_num) / self.frame_rate
                        speed_mps = distance_covered / time_elapsed if time_elapsed > 0 else 0
                        speed_kmph = speed_mps * 3.6

                        if object_type not in total_distance:
                            total_distance[object_type] = {}

                        if tracker_id not in total_distance[object_type]:
                            total_distance[object_type][tracker_id] = 0

                        total_distance[object_type][tracker_id] += distance_covered

                        # Update metrics for each frame in the batch
                        for frame_num_batch in range(frame_num, last_frame):
                            if tracker_id not in object_tracks[object_type][frame_num_batch]:
                                continue

                            # Store metrics
                            object_tracks[object_type][frame_num_batch][tracker_id]['speed'] = speed_kmph
                            object_tracks[object_type][frame_num_batch][tracker_id]['distance'] = total_distance[object_type][tracker_id]

                            # Stride rate metrics
                            stride_length = 1.5  # Average stride length in meters
                            stride_rate = (speed_mps / stride_length) * 60 if stride_length > 0 else 0
                            object_tracks[object_type][frame_num_batch][tracker_id]['stride_rate'] = stride_rate

        self.tracks = tracks
        self.metrics = total_distance