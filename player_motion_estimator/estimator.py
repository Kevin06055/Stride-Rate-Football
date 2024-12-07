import cv2
from utility import measure_distance

class GaitMetricsEstimator:
    """
    Class for estimating gait metrics like speed, distance, and stride rate for tracked objects.
    """
    def __init__(self, frame_rate=24, stride_length=1.5, meters_per_pixel=0.0005, max_speed_mps=500):
        """
        Initialize the estimator with frame rate, average stride length, pixel-to-meter scaling, and normalization option.

        :param frame_rate: Frames per second for video processing.
        :param stride_length: Average stride length (in meters).
        :param meters_per_pixel: Conversion factor from pixels to meters.
        :param max_speed_mps: Max speed threshold to avoid inflated stride rates.
        """
        self.frame_window = 5  # Sliding window for speed estimation
        self.frame_rate = frame_rate
        self.stride_length = stride_length
        self.meters_per_pixel = meters_per_pixel
        self.max_speed_mps = max_speed_mps  # Maximum speed to limit unrealistic strides

        self.tracks = {}
        self.metrics = {}
        self.total_distance = {}  # Persistent total distance storage

        # Normalization parameters with realistic ranges for stride rate
        self.stride_rate_min = 10  # Min stride rate in steps/min
        self.stride_rate_max = 200  # Max stride rate in steps/min

    def clamp_stride_rate(self, stride_rate):
        """
        Clamp the stride rate between 10 and 200 steps per minute.
        
        :param stride_rate: Stride rate in steps per minute.
        :return: Clamped stride rate (between 10 and 200).
        """
        # Clamp stride rate between 10 and 200 steps per minute
        stride_rate = max(self.stride_rate_min, min(stride_rate, self.stride_rate_max))
        return stride_rate

    def add_metrics(self, tracks):
        """
        Add estimated metrics (speed, distance, stride rate) to the track information.

        :param tracks: Dictionary containing the tracked objects.
        """
        for object_type, object_tracks in tracks.items():
            if object_type not in ["players", "goalkeepers"]:
                continue

            for tracker_id, track_data in object_tracks.items():
                if tracker_id not in self.total_distance:
                    self.total_distance[tracker_id] = 0

                frame_keys = sorted(track_data.keys())
                for frame_idx in range(0, len(frame_keys), self.frame_window):
                    frame_num = frame_keys[frame_idx]
                    end_frame_num = min(frame_keys[-1], frame_num + self.frame_window)

                    start_position = track_data.get(frame_num, {}).get('position_transformed')
                    end_position = track_data.get(end_frame_num, {}).get('position_transformed')

                    if not start_position or not end_position:
                        continue

                    # Convert distance from pixels to meters
                    distance_covered = measure_distance(start_position, end_position) * self.meters_per_pixel
                    time_elapsed = (end_frame_num - frame_num + 1) / self.frame_rate
                    speed_mps = distance_covered / time_elapsed if time_elapsed > 0 else 0

                    # Limit speed to max speed threshold to avoid inflated stride rates
                    speed_mps = min(speed_mps, self.max_speed_mps)

                    speed_kmph = speed_mps * 3.6  # Speed in km/h

                    self.total_distance[tracker_id] += distance_covered

                    # Calculate stride rate in steps per minute
                    stride_rate = (speed_mps / self.stride_length) * 60  # Stride rate in steps per minute

                    # Clamp the stride rate between 10 and 200 steps/min
                    stride_rate = self.clamp_stride_rate(stride_rate)

                    # Add metrics to the track data for the current window
                    for frame in range(frame_num, end_frame_num + 1):
                        if frame not in track_data:
                            continue

                        track_data[frame].setdefault('metrics', {})
                        track_data[frame]['metrics']['speed_kmph'] = speed_kmph
                        track_data[frame]['metrics']['distance'] = self.total_distance[tracker_id]
                        track_data[frame]['metrics']['stride_rate'] = stride_rate

        self.tracks = tracks
        self.metrics = self.total_distance

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
