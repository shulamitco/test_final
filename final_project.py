import cv2
import numpy as np
from matplotlib import pyplot as plt
# import argparse
# from PIL import Image, ImageDraw
# from math import sqrt, pi, cos, sin, atan2
# from collections import defaultdict
# from PIL import Image, ImageChops

def crop_imges_1(img):
    org = img.copy()
    kernel = np.ones((5,5),np.uint8)
    img = cv2.medianBlur(img,5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.erode(gray, kernel, iterations=20)
    gray = cv2.dilate(gray, kernel, iterations=25)
    _, gray = cv2.threshold(gray,40,255,cv2.THRESH_BINARY)
    gray = cv2.erode(gray, kernel, iterations=50)
    gray = cv2.dilate(gray, kernel, iterations=100)
    gray = cv2.erode(gray, kernel, iterations=100)
    dst = cv2.cornerHarris(gray,70,3,0.22)
    list_tf = [dst>0.02*dst.max()]
    points_indexes = np.argwhere(list_tf)
    y, x = np.swapaxes(points_indexes,0,1)[1],np.swapaxes(points_indexes,0,1)[2]
    y_max = y.max()
    y_min = y.min()
    x_size = img.shape[1]
    y_index = np.where(y == y_min)
    x_min = x[y_index].min()
    crop_img = org[0:y_max, x_min:x_size]
    return crop_img

def crop_imges_2(img):
    org = img.copy()
    img = cv2.medianBlur(img,5)
    kernel = np.ones((5,5),np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.erode(gray, kernel, iterations=20)
    gray = cv2.dilate(gray, kernel, iterations=40)
    gray = cv2.erode(gray, kernel, iterations=20)
    gray = ~gray
    gray = cv2.erode(gray, kernel, iterations=300)
    gray = cv2.dilate(gray, kernel, iterations=200)
    dst = cv2.cornerHarris(gray,40,3,0.2)
    list_tf = [dst>0.02*dst.max()]
    points_indexes = np.argwhere(list_tf)
    y, x = np.swapaxes(points_indexes,0,1)[1],np.swapaxes(points_indexes,0,1)[2]
    y = y.min()
    x_size = img.shape[1]
    y_size = img.shape[0]
    x = x.min()
    crop_img = org[y:y_size, x:x_size]
    return crop_img

def crop_imges_3(img):
    org = img.copy()
    kernel = np.ones((5,5),np.uint8)
    img = cv2.medianBlur(img,5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.erode(gray, kernel, iterations=20)
    gray = cv2.dilate(gray, kernel, iterations=40)
    gray = cv2.erode(gray, kernel, iterations=40)
    _, gray = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)
    dst = cv2.cornerHarris(gray,10,3,0.1)
    list_tf = [dst>0.02*dst.max()]
    points_indexes = np.argwhere(list_tf)
    y, x = np.swapaxes(points_indexes,0,1)[1],np.swapaxes(points_indexes,0,1)[2]
    max_y = y.max()
    x_size = img.shape[1]
    y_index = np.where(y == max_y)
    x = (x[y_index]).max()
    crop_img = org[0:max_y, x:x_size]
    return crop_img

def draw_points(img):
    img = img.copy()
    kernel = np.ones((5,5),np.uint8)
    img = cv2.medianBlur(img,5)
    img = cv2.dilate(img, kernel, iterations=3)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.float32(img)
    img = cv2.erode(img, kernel, iterations=3)
    img = cv2.erode(img, kernel, iterations=15)
    img = cv2.dilate(img, kernel, iterations=15)
    dst = cv2.cornerHarris(img,40,3,0.04)

    return dst


def activate_ex1():
    picture_path_4 = 'pictures/q1/4.JPG'
    picture_path_12 = 'pictures/q1/12.JPG'
    picture_path_13 = 'pictures/q1/13.JPG'
    img = cv2.imread(picture_path_4)
    org1 = img.copy()
    crop_img1 = crop_imges_1(img)
    img2 = cv2.imread(picture_path_12)
    org2 = img2.copy()
    crop_img2 = crop_imges_2(img2)

    img3 = cv2.imread(picture_path_13)
    org3 = img3.copy()
    crop_img3 = crop_imges_3(img3)
    dst = draw_points(~img)
    img[dst>0.02*dst.max()] = [255,0,0]
    dst = draw_points(img2)
    img2[dst>0.02*dst.max()] = [255,0,0]
    dst = draw_points(img3)
    img3[dst>0.02*dst.max()] = [255,0,0]
    titles = ["org", "pointed", "crop img"]
    pictures = [org1, img, crop_img1]

    for i in range(len(pictures)):
        plt.subplot(1,3,i+1),plt.imshow(pictures[i])
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    titles2 = ["org", "pointed", "crop img"]
    pictures2 = [org2, img2, crop_img2]

    for i in range(len(pictures2)):
        plt.subplot(1,3,i+1),plt.imshow(pictures2[i])
        plt.title(titles2[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    titles3 = ["org", "pointed", 'crop']
    pictures3 = [org3, img3, crop_img3]

    for i in range(len(pictures3)):
        plt.subplot(1,3,i+1),plt.imshow(pictures3[i], 'gray')
        plt.title(titles3[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

def find_circles(picture_path):
    # Loads an image
    src = cv2.imread(cv2.samples.findFile(picture_path), cv2.IMREAD_COLOR)
    kernel = np.ones((3,3),np.uint8)
    src = cv2.erode(src, kernel, iterations=1)
    src = cv2.dilate(src, kernel, iterations=1)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + picture_path + '] \n')
        return -1


    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=45, param2=60 ,minRadius=6, maxRadius=50)


    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, (255, 0, 255), 3)


    plt.subplot(1,1,1),plt.imshow(src)
    plt.show()
    return 0

def activate_ex2():
     for i in range(31,46):
        ex2_pic_path = 'pictures/q2/000{}.png'.format(i)
        find_circles(ex2_pic_path)


def find_the_right_pic(small_image, large_image):
    method = cv2.TM_SQDIFF_NORMED

    result = cv2.matchTemplate(small_image, large_image, method)

    # We want the minimum squared difference
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print('min - {} max - {}, rate - {}'.format(min_val,max_val,round(max_val/min_val,2)))
    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx,MPy = min_loc

    # Step 2: Get the size of the template. This is the same size as the match.
    trows,tcols = small_image.shape[:2]

    # Step 3: Draw the rectangle on large_image
    cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)
    return large_image, min_val, max_val


def find_accuracy():
    pass


def ex3_1():
    ex3_pic_path = 'pictures/q3/00031_3.png'
    img = cv2.imread(ex3_pic_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    x,y,w,h  = clean_frame(thresh)
    crop = img[y:y+h,x:x+w]
    crop = img[90:195,90:195]
    plt.subplot(111),plt.imshow(crop)
    plt.show()
    # small_image = crop[30:250,0:250]
    small_image = cv2.medianBlur(crop,3)
    # _,small_image = cv2.threshold(small_image,120,255,cv2.THRESH_BINARY)
    cv2.imshow("crop", small_image)
    cv2.waitKey()
    pic, max_val, min_val = None, 0.5, 0.5
    for i in range(31,46):
        # Read the images from the file
        large_image = cv2.imread('pictures/q2/000{}.png'.format(i))
        # large_image = cv2.medianBlur(large_image,5)

        if small_image.shape[1] > large_image.shape[1]:
            small_image = small_image[0:small_image.shape[0], 0:large_image.shape[1]]
        found_img, min_v, max_v = find_the_right_pic(small_image,large_image)
        # find_accuracy()
        # if found_img is not None:
        #     cv2.imshow("found", found_img)
        #     cv2.waitKey()


def clean_frame(thresh):
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    return cv2.boundingRect(cnt)



def ex3_2():
   pass


def activate_ex3():
    ex3_1()

if __name__ == '__main__':
    # activate_ex1()
    # activate_ex2()
    activate_ex3()



