# Number of Images of User to store
N = 50
# Python script to recognize face
import numpy as np
import cv2
import sys
from skimage.measure import structural_similarity as ssim
# Video feed
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
#
# Picture of Root User
# Crop out the Face
# Capture frame-by-frame
user = cv2.imread('jackson.jpg')
gray = cv2.cvtColor(user, cv2.COLOR_BGR2GRAY)
user_face = face_cascade.detectMultiScale(user, 1.2, 5)
for (x, y, w, h) in user_face:
    x_dim = w
    y_dim = h
    cv2.rectangle(user, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cropped =  gray[y:y+h, x:x+w]
    user_resized = cv2.resize(cropped, (200, 200))
    # Display the resulting frame
    # cv2.imshow('Root User', user_resized)


iterations = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.4, 5)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_gray = gray[y:y+h, x:x+w]
        face_resized=cv2.resize(face_gray, (200, 200))
        font = cv2.FONT_HERSHEY_PLAIN
        text = 'Scan Human'
        cv2.putText(frame,text,(50,50), font, 1,(255,255,255),2)
	cv2.imshow('Video', frame)
        scan = ssim(face_resized, user_resized)
	iter = str(iterations)
	filename = 'test' + iter + '.png'
        print scan
	if scan >=0.4:
		print_name = 'Jackson Chief Elk'
        	cv2.putText(frame,print_name,(100,100), font, 1,(255,255,255),2)
		cv2.imshow('Video', frame)
       # Write face to file
	cv2.imwrite(filename, face_resized)
	iterations+=1
        if  cv2.waitKey(1) & 0xFF == ord('q'):
            break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()