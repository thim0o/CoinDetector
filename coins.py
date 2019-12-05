import cv2 as cv
import numpy as np
import time
from copy import deepcopy
import os

# Trackbar values
max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 18
low_V = 0
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
openingAmount = 0
medianBlurAmount = 31


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


def on_opening_trackbar(val):
    global openingAmount
    openingAmount = max(1, val)

    cv.setTrackbarPos("openingAmount", window_detection_name, openingAmount)


def on_median_trackbar(val):
    global medianBlurAmount
    medianBlurAmount = max(1, val + (1 - val % 2))
    cv.setTrackbarPos("medianBlurAmount", window_detection_name, medianBlurAmount)


def getParams():
    params = cv.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 900000000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8

    return params


def setupTrackbar():
    cv.namedWindow(window_capture_name)
    cv.namedWindow(window_detection_name)

    cv.resizeWindow(window_detection_name, 600, 600)

    cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)
    cv.createTrackbar("openingAmount", window_detection_name, openingAmount, max_value, on_opening_trackbar)
    cv.createTrackbar("medianBlurAmount", window_detection_name, medianBlurAmount, max_value, on_median_trackbar)


def saveNewBgImage():
    cam = cv.VideoCapture(0)
    ret_val, raw = cam.read()
    cv.imwrite("bg.bmp", raw)


def getForeground(img, bgImg, opening, medianBlur):
    imgHsv = cv.cvtColor(img.copy(), cv.COLOR_BGR2HSV)
    bgImgHsv = cv.cvtColor(bgImg.copy(), cv.COLOR_BGR2HSV)
    difference = cv.absdiff(imgHsv, bgImgHsv)
    fgMask = cv.inRange(difference, (low_H, low_S, low_V), (high_H, high_S, high_V))

    if opening:
        kernel = cv.getStructuringElement(cv.MORPH_OPEN, tuple([openingAmount] * 2))
        fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)

    if medianBlur:
        fgMask = cv.medianBlur(fgMask, medianBlurAmount)

    foreground = cv.bitwise_and(img, img, mask=fgMask)

    return difference, fgMask, foreground


def getThresholdedBlurredImg(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, (low_H, 1, low_V), (high_H, high_S, high_V))
    im_floodfill = thresh.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = thresh | im_floodfill_inv
    im_out = cv.bitwise_not(im_out)
    im_out = cv.medianBlur(im_out, 5)

    return im_out


def resizeImg(img, resizeFactor):
    if resizeFactor == 1:
        return img
    return cv.resize(img, (0, 0), fx=resizeFactor, fy=resizeFactor)


def getCoinImages(keyPoints, raw, showCoins=True):
    images = []
    for i, kp in enumerate(keyPoints):
        x, y = kp.pt
        size = 70

        startX = int(x - size)
        stopX = int(x + size)

        startY = int(y - size)
        stopY = int(y + size)

        if startX < 0 or startY < 0 or stopX > 720 or stopY > 1280:
            print("Can't take full image of this coin!!")
            continue

        coinImg = raw[startY:stopY, startX:stopX]
        images.append(coinImg)

        if showCoins:
            cv.imshow("coin" + str(i), coinImg)

    return images


def saveImages(coinImages, folder="images", subFolder="none"):
    location = os.path.join(folder, subFolder)
    print("SAVING {} images in {}".format(len(coinImages), location))

    if not os.path.exists(location):
        os.mkdir(location)

    timeStamp = time.ctime().replace(":", "-")

    for i, img in enumerate(coinImages):
        fileName = "{} {}.bmp".format(timeStamp, i)
        fullPath = os.path.join(location, fileName)
        cv.imwrite(fullPath, img)
        print(fullPath)


def main(usingWebcam=True, newBg=False, resizeFactor=1, lotsOfPlots=True, showCoins=True, trackBar=True):
    cam = cv.VideoCapture(0)
    raw = cv.imread("bgfg.bmp")

    detector = cv.SimpleBlobDetector_create(getParams())

    if newBg:
        saveNewBgImage()

    background = cv.imread("bg.bmp")
    background = resizeImg(background, resizeFactor)

    if trackBar:
        setupTrackbar()

    while True:
        lastStart = time.time()

        if usingWebcam:
            ret_val, raw = cam.read()

        img = deepcopy(raw)
        img = resizeImg(img, resizeFactor)

        difference, mask, fg = getForeground(img, background, openingAmount, medianBlurAmount)
        threshBlurred = getThresholdedBlurredImg(fg)
        fg = cv.bitwise_and(img, img, mask=threshBlurred)

        keyPoints = detector.detect(threshBlurred)

        imgKeyPoints = cv.drawKeypoints(img, keyPoints, np.array([]), (0, 0, 255),
                                        cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        images = getCoinImages(keyPoints, raw, showCoins)

        if lotsOfPlots:
            cv.imshow('background', background)
            cv.imshow('fg', fg)
            cv.imshow('threshBlurred', threshBlurred)
            cv.imshow('foreground mask', mask)
            cv.imshow("dif", difference)

        cv.imshow('imgKeyPoints', imgKeyPoints)

        k = cv.waitKey(1)
        if k == 27:
            break  # esc to quit
        elif k == ord("s"):
            saveImages(images)

        fps = round(1 / (time.time() - lastStart), 1)
        print("{} coins detected. {} fps".format(len(keyPoints), fps))
        # time.sleep(0.05)


if __name__ == '__main__':
    import cProfile

    cProfile.run("main()")
