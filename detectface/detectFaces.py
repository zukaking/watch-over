import os
import boto3
import json
import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation
import time
import requests

my_session = boto3.Session(profile_name='plism')
client = my_session.client('rekognition',region_name = 'ap-northeast-1')

def rotate(point, theta_x, theta_y, theta_z):
    theta_x = math.pi*theta_x/180
    theta_y = math.pi*theta_y/180
    theta_z = math.pi*theta_z/180
    rot_x = np.array([[ 1,                 0,                  0],
                      [ 0, math.cos(theta_x), -math.sin(theta_x)],
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])

    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                      [                 0, 1,                  0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])

    rot_z = np.array([[ math.cos(theta_z), -math.sin(theta_z), 0],
                      [ math.sin(theta_z),  math.cos(theta_z), 0],
                      [                 0,                  0, 1]])

    rot_matrix  = rot_z.dot(rot_y.dot(rot_x))
    rot_point   = rot_matrix.dot(point.T).T
    return rot_point

def detect_face(img):
    # 3.detect face
    tmp_input_img = "images/tmp.png"
    #detect_responce = 'detect/tmp.json'
    #output_img = "images/output/tmp.png"
   
    copied_image = img.copy()
    cv2.imwrite(tmp_input_img,copied_image)
    with open(tmp_input_img, 'rb') as image:
        response = client.detect_faces(Image={'Bytes': image.read()},Attributes=['ALL'])
        #print(response)

    #with open(detect_responce, 'w') as f:
    #    print(response)
        #json.dump(response, f, indent=4)
    #continue
    

    # 4.draw face box
    #with open(detect_responce, 'r') as f:
    #    response = json.loads(f.read())

    if response["FaceDetails"]:       

        width   = response["FaceDetails"][0]["BoundingBox"]["Width"]
        height  = response["FaceDetails"][0]["BoundingBox"]["Height"]
        left    = response["FaceDetails"][0]["BoundingBox"]["Left"]
        top     = response["FaceDetails"][0]["BoundingBox"]["Top"]
        roll    = response["FaceDetails"][0]["Pose"]["Roll"]
        yaw     = response["FaceDetails"][0]["Pose"]["Yaw"]
        pitch   = response["FaceDetails"][0]["Pose"]["Pitch"]
        
        img_height, img_width, _ = img.shape
        width   = int(img_width * width)
        height  = int(img_height * height)
        left    = int(img_width * left)
        top     = int(img_height * top)
        img = cv2.rectangle(img, (left, top), (left+width, top+height), (255,0,0))

        center_x    = int(left + width/2)
        center_y    = int(top + height/2)
        x_axis      = np.array([30, 0, 0])
        y_axis      = np.array([ 0,30, 0])
        z_axis      = np.array([ 0, 0,30])
        rot_x_axis  = rotate(x_axis, roll, yaw, pitch)
        rot_y_axis  = rotate(y_axis, roll, yaw, pitch)
        rot_z_axis  = rotate(z_axis, roll, yaw, pitch)
        
        img = cv2.line(img, (center_x, center_y), (int(center_x+rot_x_axis[0]), int(center_y+rot_x_axis[1])), (0,0,255), 5)
        img = cv2.line(img, (center_x, center_y), (int(center_x+rot_z_axis[0]), int(center_y+rot_z_axis[1])), (255,0,0), 5)
        img = cv2.line(img, (center_x, center_y), (int(center_x+rot_y_axis[0]), int(center_y+rot_y_axis[1])), (0,255,0), 5)
    else:
        print("**not detect**")

    detect_angle = 0

    return detect_angle


def detect_face_down(detectAngles: list):

    ###アルゴリズム
    return False

def send_email():
    url = 'https://uxvgbs2pbg.execute-api.ap-northeast-1.amazonaws.com/test/send-sns-resoure'

    subject = "Subject test"
    message = "Message test"
    response = requests.post(url, data=json.dumps({"subject": subject, "message" : message}))
    
def capture_camera():
    
    camera_id = 0 #0:incam 1:***
    cap = cv2.VideoCapture(camera_id)

    continuous_count = 5

    detect_angles = []
    while(True):
       
        ret, frame = cap.read()
        if ret == False:
            break
        
        angle = detect_face(frame)

        if len(detect_angles) < continuous_count:
            detect_angles.append(angle)
        else:
            detect_angles.pop(0)
            detect_angles.append(angle)
        
        print(detect_angles)

        cv2.imshow("Input", frame) #ウィンドウに画像を表示する
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.5)
        
        if len(detect_angles) >= continuous_count-1 and detect_face_down(detect_angles):
            print("send email!!!!")
            send_email()
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    capture_camera()

