import supervision as sv 
from tqdm import tqdm 


def extract_crops(source_video,PLAYER_DETECTION_MODEL,PLAYER_ID,STRIDE=30):
    frame_generator = sv.get_video_frames_generator(source_path=source_video,stride=STRIDE)
    crops=[]
    for frame in tqdm(frame_generator,desc="collection frames"):
        result = PLAYER_DETECTION_MODEL.infer(frame,confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=0.5,class_agnostic=True)
        crops+=[sv.crop_image(frame,xyxy) for xyxy in detections.xyxy]

    return crops