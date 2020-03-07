from scipy.spatial import distance as dist
import imutils
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import datetime
import argparse
import time
import dlib
import cv2

def eye_aspect_ratio(eye):

#computing euclidean distances between the two sets of vertical eye landmarks (x,y) - coordinates

	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

#compute the euclidean distance between the horizontal eye landmark (x, y)- coordinates

	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear






ap = argparse.ArgumentParser()
#ap.add_argument("-p","--shape-predictor", required = True, help = "path to facial landmark predictor")
ap.add_argument("-r", "--picamera", required = False, type=int, default=-1, help = "whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate blink and then a second constant for the number of consecutive frames the eye must be below the threshold


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 4

#initialize the frame counters and total number of blinks

COUNTER = 0
TOTAL = 0

#initialize dlib's face detector (HOG based) and then create the facial landmark predictor

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/vanshika/Downloads/facial-landmarks/shape_predictor_68_face_landmarks.dat")

#grab the indexes of the facial landmarks for the left and right eye respectively

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#"initializing video stream and allow the camera sensor to warmup"

print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

#loop over the frames from the video stream

while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=450)

	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray,0)


	for rect in rects:

		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		ear = (leftEAR +rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)



		if ear < EYE_AR_THRESH:
			COUNTER +=1
		else:
			if COUNTER >=EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

	

			cv2.putText(frame, "Blinks: {}".format(TOTAL), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
			COUNTER = 0

		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

		cv2.putText(frame, "Counter: {:.2f}".format(COUNTER), (300,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

	

		

		
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
	
		break

	

cv2.destroyAllWindows()
vs.stop()




