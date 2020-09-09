import cv2
import numpy as np


def detect_pupil(img):
    pupil_center = [None, None]
    pupil_radius = None

    inv = cv2.bitwise_not(img)
    try:
        thresh = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    except BaseException:
        return pupil_center, pupil_radius

    kernel = np.ones((4, 4), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    ret, thresh1 = cv2.threshold(erosion, 220, 255, cv2.THRESH_BINARY)
    cnts, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) != 0:
        c = max(cnts, key=cv2.contourArea)
        (x, y), pupil_radius = cv2.minEnclosingCircle(c)
        pupil_center = (int(x), int(y))
        pupil_radius = int(pupil_radius)
        # cv2.circle(img, pupil_center, pupil_radius, (255, 0, 0), 2)

    return pupil_center, pupil_radius


def detect_iris(img, pupil):
    if pupil[0] is None or pupil[1] is None or pupil[2] is None:
        return None

    _, t = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)

    try:
        c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, pupil[2] * 2, param2=150)
    except BaseException:
        return None

    if c is None:
        return None

    # Find the iris using the radius of the pupil as input.
    for l in c:
        for circle in l:
            center = (pupil[0], pupil[1])
            radius = int(circle[2])
            # This creates a black image and draws an iris-sized white circle in it.
            mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
            cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)
            # Mask the iris and crop everything outside of its radius.
            img = cv2.bitwise_and(img, mask)

    return img


if __name__ == '__main__':
    img = cv2.imread('test eye.jpg', 1)
    center, radius = detect_pupil(img)
    iris_img = detect_iris(img, (center[0], center[1], radius))
    cv2.imshow('my isris', iris_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
