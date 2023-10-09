import json
import base64
import cv2
import numpy as np

HAAR_FILE = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAAR_FILE)

def base64_to_cv2(image_base64):
    image_bytes = base64.b64decode(image_base64)
    np_array = np.fromstring(image_bytes, np.uint8)
    image_cv2 = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image_cv2


def handler(event, context):
    image_data = base64_to_cv2(event['file'])
    face = cascade.detectMultiScale(image_data)
    return {
        'statusCode': 200,
        'body': json.dumps(face.tolist())
    }