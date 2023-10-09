import cv2 as cv

HAAR_FILE = "haarcascade_frontalface_default.xml"
cascade = cv.CascadeClassifier(HAAR_FILE)

camera_id = 0 #0:incam 1:***
cap = cv.VideoCapture(camera_id)

while(True):
    ret, frame = cap.read()
    
    face = cascade.detectMultiScale(frame)

    print(face)

    for x, y, w, h in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)

    cv.imshow('Capture',frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    break

cap.release()
cv.destroyAllWindows()