import os
import boto3
import json
import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation

client = boto3.client('rekognition', 'ap-northeast-1')

def rotate(point, theta_x, theta_y, theta_z):
    # theta_x = math.radians(theta_x)
    # theta_y = math.radians(theta_y)
    # theta_z = math.radians(theta_z)
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

if __name__=="__main__":
    age = 0
    people = 0

    with open("baby_orogin.png", 'rb') as image:
        response = client.detect_faces(Image={'Bytes': image.read()},Attributes=['ALL'])

    width   = response["FaceDetails"][0]["BoundingBox"]["Width"]
    height  = response["FaceDetails"][0]["BoundingBox"]["Height"]
    left    = response["FaceDetails"][0]["BoundingBox"]["Left"]
    top     = response["FaceDetails"][0]["BoundingBox"]["Top"]

    roll    = response["FaceDetails"][0]["Pose"]["Roll"]
    yaw     = response["FaceDetails"][0]["Pose"]["Yaw"]
    pitch   = response["FaceDetails"][0]["Pose"]["Pitch"]

    img = cv2.imread("baby.jpg")
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
    img = cv2.line(img, (center_x, center_y), (int(center_x+rot_x_axis[0]), int(center_y+rot_x_axis[1])), (0,0,255))
    img = cv2.line(img, (center_x, center_y), (int(center_x+rot_y_axis[0]), int(center_y+rot_y_axis[1])), (0,255,0))
    img = cv2.line(img, (center_x, center_y), (int(center_x+rot_z_axis[0]), int(center_y+rot_z_axis[1])), (255,0,0))

    cv2.imwrite("baby_result.jpg", img)

    client = boto3.client('s3',region_name='ap-northeast-1')
    client.upload_file("baby_result.jpg", "plism-spa", "baby_resutl.jpg")