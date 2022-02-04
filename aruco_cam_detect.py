#!/usr/bin/env python

# python detect_aruco_image.py --image images/example_01.png --type DICT_4X4_100

# import the necessary packages
import argparse
import imutils
import cv2
import sys
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	default="DICT_4X4_100",
	help="type of ArUCo tag to detect")
ap.add_argument("-i", "--iterations", type=int,
	default=1,
	help="Iterations count")

args = vars(ap.parse_args())

# open camera
videoCapture = cv2.VideoCapture(0)
if not videoCapture.isOpened():
    print("[ERROR]Cannot open camera")
    exit()

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000
}


# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
	sys.exit(0)

# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()


iterationsToDo = args["iterations"]

while (iterationsToDo > 0) :
	print("[INFO] Iterations to do {}".format(iterationsToDo))
	iterationsToDo = iterationsToDo - 1

	# capture input image from camera and resize it
	print("[INFO] Capturing image...")

	timeStamp0 = time.perf_counter()

	success, image = videoCapture.read()
	if not success:
		print("[ERROR]Cannot capture image from camera")
		sys.exit(0)

	#success = cv2.imwrite("filename-raw.jpg",image)
	#image = imutils.resize(image, width=600)

	(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
		parameters=arucoParams)

	# verify *at least* one ArUco marker was detected
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()

		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			# draw the bounding box of the ArUCo detection
			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

			# draw the ArUco marker ID on the image
			cv2.putText(image, str(markerID),
				(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[INFO] ArUco marker ID: {}".format(markerID))


	timeStamp1 = time.perf_counter()
	timeConsumed = timeStamp1 - timeStamp0
	print("[INFO] consumed {}".format(timeConsumed))


# write output image
print("[INFO] write result image...")
success = cv2.imwrite("filename-result.jpg", image)

videoCapture.release()