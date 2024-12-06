import supervision as sv
from collections import defaultdict
from model import player, field
from team_preprocessor import extract_crops
from team_classifier import TeamClassifier
from Persepctive_Transformer import ViewTransformer
from PitchConfig import SoccerPitchConfiguration
from PitchAnnotators import draw_pitch, draw_points_on_pitch
from player_motion_estimator import GaitMetricsEstimator
from Goalkeeper_resolver import resolve_goalkeepers_team_id
from utility import get_foot_position
import cv2
import numpy as np
import os
from tqdm import tqdm
import warnings
import numpy as np

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

def create_stats_overlay(frame, players_stats):
    """
    Create a stats overlay box in the top right corner of the frame.
    
    Args:
    frame (np.ndarray): Input frame
    players_stats (dict): Dictionary of player stats
    
    Returns:
    np.ndarray: Frame with stats overlay
    """
    # Define overlay parameters
    overlay_width = 250
    overlay_height = len(players_stats) * 30 + 40
    overlay_margin = 10
    
    # Create a semi-transparent overlay
    overlay = frame.copy()
    overlay_rect = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
    
    # Fill with semi-transparent white background
    cv2.rectangle(overlay_rect, (0, 0), (overlay_width, overlay_height), 
                  (255, 255, 255), -1)
    cv2.addWeighted(overlay_rect, 0.7, np.zeros_like(overlay_rect), 0.3, 0, overlay_rect)
    
    # Title
    cv2.putText(overlay_rect, "Player Stride Rates", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 0), 2)
    
    # Add player stats
    for i, (tracker_id, stride_rate) in enumerate(players_stats.items()):
        text = f"Player #{tracker_id}: {stride_rate:.2f} strides/s"
        cv2.putText(overlay_rect, text, 
                    (10, 60 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1)
    
    # Position overlay in top right corner
    x_offset = frame.shape[1] - overlay_width - overlay_margin
    y_offset = overlay_margin
    
    # Blend overlay with frame
    frame[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = overlay_rect
    
    return frame

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
            
            foot_position = get_foot_position(bbox)
            
            # Use tracker_id as the key to ensure consistent tracking
            tracks['players'][frame_num][tracker_id] = {
                'position_transformed': foot_position,
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
            
            # Use average stride rate if available
            avg_stride_rate = sum(stride_rates) / len(stride_rates) if stride_rates else 0.0
            player_stats[tracker_id] = avg_stride_rate

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

        # Create radar pitch and plot points
        radar_pitch_frame = draw_pitch(CONFIG)
        radar_pitch_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=20,
            pitch=radar_pitch_frame)
        radar_pitch_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 0],
            face_color=sv.Color.from_hex('00BFFF'),
            edge_color=sv.Color.BLACK,
            radius=26,
            pitch=radar_pitch_frame)
        radar_pitch_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 1],
            face_color=sv.Color.from_hex('FF1493'),
            edge_color=sv.Color.BLACK,
            radius=26,
            pitch=radar_pitch_frame)
        radar_pitch_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_referees_xy,
            face_color=sv.Color.from_hex('FFD700'),
            edge_color=sv.Color.BLACK,
            radius=26,
            pitch=radar_pitch_frame)

        # Resize radar pitch to fit in the video frame
        radar_pitch_frame_resized = cv2.resize(radar_pitch_frame, (frame.shape[1] // 3, frame.shape[0] // 3))

        # Calculate the position for the radar pitch at the bottom center
        overlay_position_x = (frame.shape[1] - radar_pitch_frame_resized.shape[1]) // 2
        overlay_position_y = frame.shape[0] - radar_pitch_frame_resized.shape[0] - 10  # 10px from the bottom

        # Overlay the radar pitch onto the annotated video frame
        annotated_frame[
            overlay_position_y:overlay_position_y + radar_pitch_frame_resized.shape[0],
            overlay_position_x:overlay_position_x + radar_pitch_frame_resized.shape[1]
        ] = radar_pitch_frame_resized

        # Visualize the annotated frame
        video_sink.write_frame(annotated_frame)