import json
import requests
import base64
import cv2

def cv2_to_base64(image_cv2):
    image_bytes = cv2.imencode('.jpg', image_cv2)[1].tostring()
    image_base64 = base64.b64encode(image_bytes).decode()
    return image_base64

path = "mihira.jpg"
data = cv2.imread(path)
encoded_data = cv2_to_base64(data)
url = 'https://sl9b1kdwel.execute-api.ap-northeast-1.amazonaws.com/beta' 
response = requests.post(url, data=json.dumps({'file': encoded_data}))

json_data = response.json()

print(json_data)