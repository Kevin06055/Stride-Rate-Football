import cv2
import numpy as np
import supervision as sv
from PitchAnnotators import draw_pitch,draw_points_on_pitch

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
    overlay_width = 275
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
    for i, (tracker_id, stats) in enumerate(players_stats.items()):
        text = (f"Player #{tracker_id}: "
                f"Stride: {stats['avg_stride_rate']:.2f}/min"
        )
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

def radar_pitch_overlay(pitch_ball_xy,PLAYER_DETECTION,pitch_players_xy,pitch_refrees_xy,CONFIG,FRAME,ANNOTATED_FRAME):
    radar_pitch_frame = draw_pitch(CONFIG)
    radar_pitch_frame = draw_points_on_pitch(
        config = CONFIG,
        xy=pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=20,
        pitch=radar_pitch_frame
    )
    radar_pitch_frame = draw_points_on_pitch(
        config = CONFIG,
        xy=pitch_players_xy[PLAYER_DETECTION.class_id==0],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.BLACK,
        radius=24,
        pitch=radar_pitch_frame
    )
    radar_pitch_frame = draw_points_on_pitch(
        config = CONFIG,
        xy=pitch_players_xy[PLAYER_DETECTION.class_id==1],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.BLACK,
        radius=24,
        pitch=radar_pitch_frame
    )
    radar_pitch_frame = draw_points_on_pitch(
        config = CONFIG,
        xy=pitch_refrees_xy,
        face_color=sv.Color.from_hex('FFD700'),
        edge_color=sv.Color.BLACK,
        radius=24,
        pitch=radar_pitch_frame
    )

    radar_pitch_frame_resized = cv2.resize(radar_pitch_frame, (FRAME.shape[1] // 3, FRAME.shape[0] // 3))

    # Calculate the position for the radar pitch at the bottom center
    overlay_position_x = (FRAME.shape[1] - radar_pitch_frame_resized.shape[1]) // 2
    overlay_position_y = FRAME.shape[0] - radar_pitch_frame_resized.shape[0] - 10  # 10px from the bottom

        # Overlay the radar pitch onto the annotated video frame
    ANNOTATED_FRAME[
            overlay_position_y:overlay_position_y + radar_pitch_frame_resized.shape[0],
            overlay_position_x:overlay_position_x + radar_pitch_frame_resized.shape[1]
        ] = radar_pitch_frame_resized
    return ANNOTATED_FRAME




    


