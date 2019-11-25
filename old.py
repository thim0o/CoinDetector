import cv2 as cv
import numpy as np
import time
from copy import deepcopy
from pprint import pprint


def main():
    raw = cv.imread("euros.bmp")
    img = deepcopy(raw)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    thresh1 = cv.inRange(hsv, (0, 0, 58), (180, 46, 125))
    thresh2 = cv.medianBlur(thresh1, 21)

    params = cv.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 90000000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    detector = cv.SimpleBlobDetector_create(params)

    # Detect the blobs in the image
    keypoints = detector.detect(thresh2)
    print(len(keypoints))

    imgKeyPoints = cv.drawKeypoints(raw, keypoints, np.array([]), (0, 0, 255),
                                    cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    while True:
        # cv.imshow('gray', gray)
        # cv.imshow('hsv', hsv)
        # cv.imshow('thresh', thresh)
        # cv.imshow('thresh1', thresh1)
        # cv.imshow('thresh2', thresh2)
        # cv.imshow('gray_blurred', gray_blurred)
        cv.imshow('img  with circles2', imgKeyPoints)

        if cv.waitKey(1) == 27:
            break  # esc to quit
        time.sleep(0.1)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
