import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import os


def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Customo format as per our parameters
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')

    # Define 6 x 6 format with 256 possibilities
    # By default:
    # arucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    # Making it dynamic
    arucoDict = aruco.Dictionary_get(key)

    arucoParam = aruco.DetectorParameters_create()
    # Detect markers and return bounding boxes, ID and rejected markers (found marker but no ID)
    # detectMarkers(image, dictionary, corners=None, ids=None, parameters)
    bboxes, ids, rejected = aruco.detectMarkers(
        gray, arucoDict, parameters=arucoParam)

    # print(ids)
    # print(bbox)
    if draw:
        aruco.drawDetectedMarkers(img, bboxes)


def main():
    # Get video
    capture = cv.VideoCapture(0)

    while True:
        isTrue, frame = capture.read()
        findArucoMarkers(frame)
        cv.imshow('Image', frame)

        cv.waitKey(1)


if __name__ == "__main__":
    main()
