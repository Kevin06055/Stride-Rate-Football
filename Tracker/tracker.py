import supervision as sv
def extract_tracks(frames, all_detections):
    tracks={
        "player":{},
        "goalkeeper":{},
        "refree":{}
    }


