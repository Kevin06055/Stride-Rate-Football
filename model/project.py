import os
from roboflow import Roboflow
from inference import get_model
from dotenv import load_dotenv
load_dotenv()
ROBOFLOW_API_KEY = os.getenv('ROBOLOW_API_KEY')

def player():
    PLAYER_MODEL_DETECTION_ID= 'stride-rate/4'
    PLAYER_DETECTION_MODEL = get_model(model_id = PLAYER_MODEL_DETECTION_ID,api_key = ROBOFLOW_API_KEY)

    return PLAYER_DETECTION_MODEL


def field():
    FIELD_DETECTION_MODEL_ID='football-field-detection-f07vi/14'
    FIELD_DETECTION_MODEL = get_model(FIELD_DETECTION_MODEL_ID,ROBOFLOW_API_KEY)
    
    return FIELD_DETECTION_MODEL