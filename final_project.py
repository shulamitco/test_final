import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import hypot


def find_circles(src):
    org = src.copy()
    kernel = np.ones((3,3),np.uint8)
    src = cv2.erode(src, kernel, iterations=1)
    src = cv2.dilate(src, kernel, iterations=1)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=45, param2=60 ,minRadius=6, maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(org, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(org, center, radius, (255, 0, 255), 3)



    return org
