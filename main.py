import supervision as sv
from collections import defaultdict
from model import player, field
from team_preprocessor import extract_crops
from team_classifier import TeamClassifier
from Persepctive_Transformer import ViewTransformer
from PitchConfig import SoccerPitchConfiguration
from PitchAnnotators import draw_pitch, draw_points_on_pitch
from player_motion_estimator import gait_metrics_estimator
from Goalkeeper_resolver import resolve_goalkeepers_team_id
from utility import get_foot_position
import cv2
import numpy as np
import os
from tqdm import tqdm
import warnings
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
ESTIMATOR = gait_metrics_estimator(frame_rate=fps)

# Classifier fitting for team assignment
crops = extract_crops(SOURCE_VIDEO_PATH, PLAYER_DETECTION_MODEL=PLAYER_DETECTION_MODEL, PLAYER_ID=PLAYER_ID, STRIDE=STRIDE)
team_classifier = TeamClassifier(device=DEVICE)
team_classifier.fit(crops)

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=20)

# Tracker Initialization
tracker = sv.ByteTrack()
tracker.reset()

# Frame generator and videoSink Initialization using SuperVision
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info)

tracks = {
    'players': defaultdict(dict),
    'goalkeepers': defaultdict(dict),
    'referee': defaultdict(dict)
}

# Processing frames
with video_sink:
    for frame_num,frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

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
        if frame_num not in tracks['players']:
            tracks['players'][frame_num]= {}

        
        for tracker_id, bbox in zip(players_detections.tracker_id,players_detections.xyxy):
            if tracker_id not in tracks['players'][frame_num]:
                tracks['players'][frame_num][tracker_id] = {}
            foot_position = get_foot_position(bbox)
            frame_number = frame_num
            tracks['players'][frame_num][tracker_id]['position_transformed'] = foot_position
        # Call the gait_metrics_estimator here to add speed, distance, and stride_rate
        ESTIMATOR.add_metrics(tracks=tracks)

        # Annotating the frame with gait metrics (speed, distance, stride_rate)
        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections)

        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections)
        print(tracks)
        
        # Draw labels for players and goalkeepers with gait metrics
        labels = [
            f"#{tracker_id} - Speed: {tracks['players'][tracker_id].get('speed', '1.00')} km/h\n"
            f"Distance: {tracks['players'][tracker_id].get('distance', '0.00')} m\n"
            f"Stride Rate: {tracks['players'][tracker_id].get('stride_rate', '0.oo')} strides/s"
            for tracker_id in all_detections.tracker_id
        ]
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections,
            labels=labels)

        # Annotate the frame with team information (as already implemented)
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

        # Write annotated frame to video
        video_sink.write_frame(annotated_frame)
