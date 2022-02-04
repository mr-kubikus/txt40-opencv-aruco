#!/usr/bin/env python

# python detect_aruco_image.py --image images/example_01.png --type DICT_4X4_100

# import the necessary packages
import argparse
import imutils
import cv2
import sys
import time

def load_coefficients(path):
	""" Loads camera matrix and distortion coefficients. """
	# FILE_STORAGE_READ
	cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
	
	if not cv_file:
		print("[ERROR]Cannot open Camera parameters file")
		exit()
	# note we also have to specify the type to retrieve other wise we only get a
	# FileNode object back instead of a matrix
	camera_matrix = cv_file.getNode("K").mat()
	dist_matrix = cv_file.getNode("D").mat()
	print(camera_matrix)
	print(dist_matrix)
	cv_file.release()
	return [camera_matrix, dist_matrix]
	
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	default="DICT_4X4_100",
	help="type of ArUCo tag to detect")
ap.add_argument("-i", "--iterations", type=int,
	default=1,
	help="Iterations count")
ap.add_argument("-c", "--camera_params_file", type=str, required=True, help='Camera parameters file')

args = vars(ap.parse_args())

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

cameraMatrix, distMatrix = load_coefficients(args["camera_params_file"])

# open camera
videoCapture = cv2.VideoCapture(0)
if not videoCapture.isOpened():
    print("[ERROR]Cannot open camera")
    exit()

videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

	#image = imutils.resize(image, width=600)

	(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
		parameters=arucoParams, cameraMatrix=cameraMatrix, distCoeff=distMatrix)

	# verify *at least* one ArUco marker was detected
	if len(ids) > 0:
		imgage = cv2.aruco.drawDetectedMarkers(image, corners, ids, (0,255,0))
		for i in range(0, len(ids)):
			
			rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 24.3, cameraMatrix, distMatrix)
			imgage = cv2.aruco.drawAxis(image, cameraMatrix, distMatrix, rvec, tvec, 25)
			print("[INFO] ArUco marker ID: {}".format(ids[i]))

	timeStamp1 = time.perf_counter()
	timeConsumed = timeStamp1 - timeStamp0
	print("[INFO] consumed {}".format(timeConsumed))


# write output image
print("[INFO] write result image...")
success = cv2.imwrite("filename-result.jpg", image)

videoCapture.release()