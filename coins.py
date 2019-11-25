import cv2 as cv
import numpy as np
import time
from copy import deepcopy

# Trackbar values
max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 1
high_H = 180
high_S = 255
high_V = 255
window_capture_name = 'Window'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


# Trackbar functions
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)


def getParams():
    params = cv.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 90000000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.7

    return params


def setupTrackbar():
    cv.namedWindow(window_capture_name)
    cv.namedWindow(window_detection_name)
    cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)


def saveNewBgImage():
    cam = cv.VideoCapture(0)
    cv.imwrite(cam.read(0), "bg.jpg")


def getForeground(img, bgImg, opening=3, medianBlur=7):
    bgSub.apply(bgImg)
    fgMask = bgSub.apply(img)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, tuple([opening] * 2))
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
    fgMask = cv.medianBlur(fgMask, medianBlur)
    foreground = cv.bitwise_and(img, img, mask=fgMask)

    return fgMask, foreground


def getThresholdedBlurredImg(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    threshBlurred = cv.medianBlur(thresh, 7)
    threshBlurred = cv.bitwise_not(threshBlurred)
    return threshBlurred


def resizeImg(img, resizeFactor):
    return cv.resize(img, (0, 0), fx=resizeFactor, fy=resizeFactor)


def main(usingWebcam=False, newBg=False, resizeFactor=0.5):
    cam = cv.VideoCapture(0)
    raw = cv.imread("bgfg.jpg")

    detector = cv.SimpleBlobDetector_create(getParams())
    setupTrackbar()

    if newBg:
        saveNewBgImage()

    background = cv.imread("bg.jpg")
    background = resizeImg(background, resizeFactor)

    while True:
        lastStart = time.time()

        if usingWebcam:
            ret_val, raw = cam.read()

        img = deepcopy(raw)
        img = resizeImg(img, resizeFactor)

        mask, fg = getForeground(img, background)
        threshBlurred = getThresholdedBlurredImg(fg)

        keyPoints = detector.detect(threshBlurred)

        imgKeyPoints = cv.drawKeypoints(img, keyPoints, np.array([]), (0, 0, 255),
                                        cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        imgKeyPoints2 = cv.drawKeypoints(fg, keyPoints, np.array([]), (0, 0, 255),
                                         cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv.imshow('threshBlurred', threshBlurred)
        cv.imshow('imgKeyPoints', imgKeyPoints)
        cv.imshow('imgKeyPoints2', imgKeyPoints2)
        cv.imshow('raw', raw)
        cv.imshow('mask', mask)

        if cv.waitKey(1) == 27:
            break  # esc to quit

        fps = round(1 / (time.time() - lastStart), 1)
        print("{} coins detected. {} fps".format(len(keyPoints), fps))
        # time.sleep(0.05)


if __name__ == '__main__':
    bgSub = cv.createBackgroundSubtractorMOG2(detectShadows=False, history=1)
    main()
