import cv2
from picamera2 import Picamera2
import time
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = Picamera2()
cam.start()
time.sleep(2)


    
def detect_faces(frame):
    # Convert the frame to grayscale, as Haar Cascade works on grayscale images
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame
def main():
    while True:
        frame = cam.capture_array()
        frame_with_faces = detect_faces(frame)
        
        # Display the frame with detected faces
        cv2.imshow("Face Detection", frame_with_faces)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cam.close()
    cv2.destroyAllWindows()

main()