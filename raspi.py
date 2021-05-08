
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

from time import sleep
from gpiozero import Buzzer
buzzer = Buzzer(18)

def euclidean_dist(ptA, ptB):
	
	return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
	
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])

	C = euclidean_dist(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")

args = vars(ap.parse_args())

ARM = 0.300
ARM_timr = 9

COUNTER = 0

print("deteksi facial landmark predictor...")
detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("mulai video stream")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	for (x, y, w, h) in rects:
		
		rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))


		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		
		ear = (leftEAR + rightEAR) / 2.0

		
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		
		if ear < ARM:
			COUNTER += 1

			
			if COUNTER >= ARM_timr:
			
					buzzer.on()
					sleep(1)
					buzzer.off()
					sleep(0.2)

				cv2.putText(frame, "Peringatan", (10, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					

		else:
			COUNTER = 0

		cv2.putText(frame, "Ear(ambang batas)= {:.3f}".format(ear), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	cv2.imshow("raspberrypi", frame)
	key = cv2.waitKey(1) & 0xFF


	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
