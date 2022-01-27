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
    # if (bboxes):
    # x, y of first corner
    # print(bboxes[0][0][0])

    if draw:
        aruco.drawDetectedMarkers(img, bboxes)

    return [bboxes, ids]


def augmentAruco(bboxes, id, img, imgAugment, drawId=True):

    topLeft = bboxes[0][0][0], bboxes[0][0][1]
    topRight = bboxes[0][1][0], bboxes[0][1][1]
    bottomRight = bboxes[0][2][0], bboxes[0][2][1]
    bottomLeft = bboxes[0][3][0], bboxes[0][3][1]

    # Get height, width, channel of image to be augmented
    height, width, channel = imgAugment.shape

    # Now,
    # we have to find the homography
    # that will give us a matrix
    # the matrix will help us to transform from one image to another
    # this will give us the ability to overlay our image at the correct position
    # on our destination image

    # Image Transformation and Perspective Transformation - read up on this to understand it
    pts1 = np.array([topLeft, topRight, bottomRight, bottomLeft])
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix, _ = cv.findHomography(pts2, pts1)

    # Warp Image
    imgOut = cv.warpPerspective(
        imgAugment, matrix, (img.shape[1], img.shape[0]))

    # Fill original image where there are markers with black
    cv.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))

    # Now, we can combine the images
    imgOut = img + imgOut

    return imgOut


def main():
    # Get video
    capture = cv.VideoCapture(0)

    while True:
        isTrue, frame = capture.read()
        # Part 1: Find aruco markers and return bounding boxes and IDs
        arucoFound = findArucoMarkers(frame)

        # Part 2: Augment something on the markers
        imgAug = cv.imread('./Images/nyu.png')

        # Loop through all markers and augment each one
        # Check if bounding box list is not empty
        if (len(arucoFound[0]) != 0):
            # Loop through both Bboxes, ID at the same time
            for bboxes, id in zip(arucoFound[0], arucoFound[1]):
                # print(bboxes, id)

                # Actual augment
                frame = augmentAruco(bboxes, id, frame, imgAug)

        cv.imshow('Image', frame)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
