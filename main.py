import supervision as sv
import argparse
from collections import defaultdict
from model import player, field
from team_preprocessor import extract_crops
from team_classifier import TeamClassifier
from Persepctive_Transformer import ViewTransformer
from PitchConfig import SoccerPitchConfiguration
from PitchAnnotators import draw_pitch, draw_points_on_pitch
from player_motion_estimator import GaitMetricsEstimator
from Goalkeeper_resolver import resolve_goalkeepers_team_id
from utility import get_center_of_boxes
import cv2
import numpy as np
import os
from tqdm import tqdm
import warnings
import numpy as np
from overlay_estimator import create_stats_overlay,radar_pitch_overlay

warnings.filterwarnings("ignore")
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration/Settings
SOURCE_VIDEO_PATH = '121364_0.mp4'
TARGET_VIDEO_PATH = 'test.mp4'
PLAYER_DETECTION_MODEL = player()
FIELD_DETECTION_MODEL = field()
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
DEVICE = 'cuda'
STRIDE = 30
CONFIG = SoccerPitchConfiguration()
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
fps = video_info.fps
ESTIMATOR = GaitMetricsEstimator(frame_rate=fps)

# Classifier fitting for team assignment
crops = extract_crops(SOURCE_VIDEO_PATH, PLAYER_DETECTION_MODEL=PLAYER_DETECTION_MODEL, PLAYER_ID=PLAYER_ID, STRIDE=STRIDE)
team_classifier = TeamClassifier(device=DEVICE)
team_classifier.fit(crops)

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=20)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)

# Tracker Initialization
tracker = sv.ByteTrack()
tracker.reset()

# Frame generator and videoSink Initialization using SuperVision
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info)

# Updated tracks structure with nested defaultdict
tracks = {
    'players': defaultdict(lambda: defaultdict(dict)),
    'goalkeepers': defaultdict(lambda: defaultdict(dict)),
    'referee': defaultdict(lambda: defaultdict(dict))
}
# Processing frames
with video_sink:
    for frame_num, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
        # Detect players 
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        # Ball and other detections processing
        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        # Separate detections by class
        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

        # Team assignment logic for players
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)

        goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
            players_detections, goalkeepers_detections)

        referees_detections.class_id -= 1

        all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])
        all_detections.class_id = all_detections.class_id.astype(int)

        # Annotate frame with labels
        labels = [
            f"#{tracker_id}"
            for tracker_id
            in all_detections.tracker_id
        ]
        annotated_frame = label_annotator.annotate(scene=frame, detections=all_detections, labels=labels)

        # Update tracks with consistent tracking
        for class_id, tracker_id, bbox in zip(all_detections.class_id, all_detections.tracker_id, all_detections.xyxy):
            if class_id == REFEREE_ID:
                continue
            
            center_of_boxes = get_center_of_boxes(bbox)  
            # Use tracker_id as the key to ensure consistent tracking
            tracks['players'][frame_num][tracker_id] = {
                'position_transformed': center_of_boxes,
                'class_id': class_id  # Store class_id for team identification
            }

        # Call gait metrics estimator
        ESTIMATOR.add_metrics(tracks=tracks)

        # Prepare player stats for overlay with robust metric retrieval
        player_stats = {}
        for tracker_id in tracks['players'][frame_num].keys():
            # Safely access metrics across all frames for this tracker_id
            stride_rates = [
                frame_tracks[tracker_id].get('metrics', {}).get('stride_rate', 0.0)
                for frame_num, frame_tracks in tracks['players'].items()
                if tracker_id in frame_tracks and 'metrics' in frame_tracks[tracker_id]
            ]

            speeds = [
                frame_tracks[tracker_id].get('metrics', {}).get('speed_kmph', 0.0)
                for frame_num, frame_tracks in tracks['players'].items()
                if tracker_id in frame_tracks and 'metrics' in frame_tracks[tracker_id]
            ]
            
            # Use average stride rate if available
            avg_stride_rate = sum(stride_rates) / len(stride_rates) if stride_rates else 0.0
            avg_speed_kmph = sum(speeds) / len(speeds) if speeds else 1
            player_stats[tracker_id] = {
                'avg_stride_rate': avg_stride_rate}

        # Add stats overlay to the frame
        annotated_frame = create_stats_overlay(annotated_frame, player_stats)
        
        # Field detection and perspective transformation
        result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result)

        # Apply perspective transformation and draw radar
        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        pitch_reference_points = np.array(CONFIG.vertices)[filter]

        transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )

        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

        players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_players_xy = transformer.transform_points(points=players_xy)

        referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_referees_xy = transformer.transform_points(points=referees_xy)
        #radar pitch generation and overlay 
        annotated_frame = radar_pitch_overlay(pitch_ball_xy=pitch_ball_xy,PLAYER_DETECTION=players_detections,pitch_players_xy=pitch_players_xy,pitch_refrees_xy=pitch_referees_xy,CONFIG=CONFIG,FRAME=frame,ANNOTATED_FRAME=annotated_frame)
        # framewriting
        video_sink.write_frame(annotated_frame)