"""
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

# this is just to unconfuse pycharm
try:
    from cv2 import cv2
except ImportError:
    pass


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)

    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)

        ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        cv2.imshow('my webcam', img)
        cv2.imshow('my webcam', thresh1)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
