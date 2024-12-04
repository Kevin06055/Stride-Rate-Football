import supervision as sv
import numpy as np 
def resolve_goalkeepers_team_id(players_detections:sv.Detections,goalkeepers_detections:sv.Detections):
    goalkeepers_xy = goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_detections.class_id==0].mean(axis=0)
    team_1_centroid = players_xy[players_detections.class_id==1].mean(axis=0)

    goalkeepers_team_ids = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy-team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy-team_1_centroid)
        goalkeepers_team_ids.append(0 if dist_0 < dist_1 else 1)

    
    return np.array(goalkeepers_team_ids)