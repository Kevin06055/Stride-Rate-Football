import supervision as sv
from collections import defaultdict
from model import player, field
from team_preprocessor import extract_crops
from team_classifier import TeamClassifier
from Persepctive_Transformer import ViewTransformer
from PitchConfig import SoccerPitchConfiguration
from utility import get_foot_position
from PitchAnnotators import draw_pitch, draw_points_on_pitch
from player_motion_estimator import gait_metrics_estimator
from Goalkeeper_resolver import resolve_goalkeepers_team_id
import cv2
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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


tracks = {
    'players':{},
    'refrees':{}
}

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info)

with video_sink:
    for frame_num ,frame in enumerate(tqdm(frame_generator,total=video_info.total_frames)):
        result  = PLAYER_DETECTION_MODEL.infer(frame,confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        ball_detections = detections[detections.class_id==BALL_ID]
        ball_detections.xyxy= sv.pad_boxes(xyxy=ball_detections.xyxy,px=10)

        all_detections = detections[detections.class_id!=BALL_ID]
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
            tracks['players'][frame_num] = {}

        
        #populates the tracks with the corresponding values

        for tracker_id, bbox in zip(players_detections.tracker_id,players_detections.xyxy):
            if tracker_id not in tracks['players'][frame_num]:
                tracks['players'][frame_num][tracker_id] = {}
            foot_position = get_foot_position(bbox)
            frame_number = frame_num
            tracks['players'][frame_num][tracker_id]['position_transformed'] = foot_position

        ESTIMATOR.add_metrics(tracks)
        



              
    