import os
import boto3
import json
import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation
import time

HAAR_FILE = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)

client = boto3.client('rekognition','ap-northeast-1')


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


def detect_face_and_writeideo(inputimg_path):
    #path = "target"
    #imageNum = 400

    path = inputimg_path
    imageNum = len(os.listdir('images/%s'%(path)))

    # 1.download
    #s3 = boto3.resource('s3')
    #bucket = s3.Bucket('watch-over-app')
    #bucket.download_file('data/IMG_3252.MOV', 'IMG_3252.MOV')
    

    # 5.upload
    """
    client = boto3.client('s3',region_name='ap-northeast-1')
    client.upload_file("%s_face.mp4"%(path), "plism-operation", "%s_face.mp4"%(path))
    a
    """

    # 2.video to images
    """   
    cap = cv2.VideoCapture("video/%s.MOV"%(path))
    i = 0
    j = 0
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        i+=1
        if i % 1 != 0:
            continue
        cv2.imwrite("images/%s/%05d.png"%(path, j), img)
        j += 1
    cap.release()
    a
    """
    
    # 4.wirte video
    
    """
    cap = cv2.VideoCapture("video/%s.MOV"%(path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)/1)
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('%s_face.mp4'%(path), fmt, fps, (width, height))
    """

    for i in range(1, imageNum, 1):
        
        # 3.detect face
        input_img = "images/%s/%05d.png"%(path, i)
        detect_responce = 'detect/%s/%05d.json'%(path, i)
        output_img = "images/output/%s/%05d.png"%(path, i)

        if not os.path.exists(input_img):
            continue

        if not os.path.exists('detect/%s'%(path)):
            os.mkdir('detect/%s'%(path))

        if not os.path.exists('images/output/%s'%(path)):
            os.makedirs('images/output/%s'%(path))
        
        with open(input_img, 'rb') as image:
            response = client.detect_faces(Image={'Bytes': image.read()},Attributes=['ALL'])
        with open(detect_responce, 'w') as f:
            print(response)
            json.dump(response, f, indent=4)
        #continue
        

        # 4.draw face box
        with open(detect_responce, 'r') as f:
            response = json.loads(f.read())

        img = cv2.imread(input_img)
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
            print(i, rot_y_axis[0], rot_y_axis[1], rot_y_axis[2])
            img = cv2.line(img, (center_x, center_y), (int(center_x+rot_x_axis[0]), int(center_y+rot_x_axis[1])), (0,0,255), 5)
            img = cv2.line(img, (center_x, center_y), (int(center_x+rot_z_axis[0]), int(center_y+rot_z_axis[1])), (255,0,0), 5)
            img = cv2.line(img, (center_x, center_y), (int(center_x+rot_y_axis[0]), int(center_y+rot_y_axis[1])), (0,255,0), 5)
        else:
            print(i)
        #writer.write(img)
        
        #cv2.imwrite(output_img,img)
        

    #writer.release()

def detect_face(img):
    # 3.detect face
    tmp_input_img = "images/tmp.png"
    detect_responce = 'detect/tmp.json'
    output_img = "images/output/tmp.png"
   
    copied_image = img.copy()
    cv2.imwrite(tmp_input_img,copied_image)
    with open(tmp_input_img, 'rb') as image:
        response = client.detect_faces(Image={'Bytes': image.read()},Attributes=['ALL'])
    with open(detect_responce, 'w') as f:
        #print(response)
        json.dump(response, f, indent=4)
    #continue
    

    # 4.draw face box
    with open(detect_responce, 'r') as f:
        response = json.loads(f.read())

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
        print("i")
    #writer.write(img)
    
    #cv2.imwrite(output_img,img)
        
def capture_camera(output_path):
    
    camera_id = 0 #0:incam 1:***
    cap = cv2.VideoCapture(camera_id)

    i = 0
    while(True):
        i+=1
        ret, frame = cap.read()
        if ret == False:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        detect_face(frame)
        
        cv2.imshow("Input", frame) #ウィンドウに画像を表示する
        
        time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    output_path  =  "IMG_2824"
    capture_camera(output_path = output_path)

