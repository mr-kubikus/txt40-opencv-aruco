#!/usr/bin/env python

# import the necessary packages
import argparse
import numpy as np
import cv2
import time
import sys


def main():
	if len(sys.argv) < 4:
		print("Usage: ./program_name directory_to_save start_index prefix")
		sys.exit(1)

	i = int(sys.argv[2])
	doCapture = True
	
	while True:
		# Capture frame-by-frame
		if doCapture:
			# Set your camera
			cap = cv2.VideoCapture(0)
			# Set these for high resolution
			# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # width
			# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # height

							
			ret, frame = cap.read()
			
			filename = sys.argv[3] + str(i) + ".png"
			cv2.imwrite(sys.argv[1] + "/" + filename, frame)
			# When everything done, release the capture
			cap.release()

			print("Captured {}".format(filename))
		
		key = input("Press 'c' for continue or 'e' to exit...")
		if key == "e":
			break
		elif key == "c":
			doCapture = True
			i += 1
		else:
			doCapture = False
			print("Use keys 'c' or 'e'");


if __name__ == '__main__':   
	main()
