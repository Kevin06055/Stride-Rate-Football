from utility import measure_distance

class GaitMetricsEstimator:
    """
    Class for estimating gait metrics like speed, distance, and stride rate for tracked objects.
    """
    def __init__(self, frame_rate=24, stride_length=1.5, meters_per_pixel=0.005, normalize=True):
        """
        Initialize the estimator with frame rate, average stride length, pixel-to-meter scaling, and normalization option.

        :param frame_rate: Frames per second for video processing.
        :param stride_length: Average stride length (in meters).
        :param meters_per_pixel: Conversion factor from pixels to meters.
        :param normalize: Whether to normalize the speed and stride rate values.
        """
        self.frame_window = 5  # Sliding window for speed estimation
        self.frame_rate = frame_rate
        self.stride_length = stride_length
        self.meters_per_pixel = meters_per_pixel
        self.normalize = normalize

        self.tracks = {}
        self.metrics = {}
        self.total_distance = {}  # Persistent total distance storage

        # Normalization parameters with more realistic ranges
        self.speed_max = 15  # Max speed in km/h for normalization (adjusted from 40)
        self.stride_rate_max = 120  # Max stride rate in steps/min for normalization (adjusted from 200)

    def normalize_values(self, speed_kmph, stride_rate):
        """
        Normalize the speed and stride rate values to a 0-1 scale using min-max normalization.

        :param speed_kmph: Speed in km/h.
        :param stride_rate: Stride rate in steps per minute.
        :return: Tuple of normalized speed and stride rate.
        """
        if self.normalize:
            # Use max of 0 to handle potential negative values
            speed_kmph = max(0, min(speed_kmph, self.speed_max)) / self.speed_max
            stride_rate = max(0, min(stride_rate, self.stride_rate_max)) / self.stride_rate_max
        
        return speed_kmph, stride_rate

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
                    speed_kmph = speed_mps * 3.6

                    self.total_distance[tracker_id] += distance_covered

                    # More accurate stride rate calculation
                    stride_rate = (speed_mps / max(self.stride_length, 0.1)) * 60 if self.stride_length > 0 else 0

                    # Normalize the values if needed
                    speed_kmph, stride_rate = self.normalize_values(speed_kmph, stride_rate)

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