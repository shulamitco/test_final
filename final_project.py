import glob

import cv2
import imagehash
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from math import hypot
from itertools import combinations

from skimage.measure import compare_ssim


def distance(p1,p2):
    """Euclidean distance between two points."""
    x1,y1 = p1
    x2,y2 = p2
    return hypot(x2 - x1, y2 - y1)

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
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
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

def find_circles(src):
    # Loads an image
    kernel = np.ones((3,3),np.uint8)
    src = cv2.erode(src, kernel, iterations=1)
    src = cv2.dilate(src, kernel, iterations=1)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        return -1


    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=45, param2=60 ,minRadius=6, maxRadius=50)
    if circles is None:
        return None

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
    images = [cv2.imread(file) for file in glob.glob("pictures/q2/*.png")]

    for img in images:
        find_circles(img)


def find_the_right_pic(small_image, large_image):
    method = cv2.TM_SQDIFF_NORMED


    result = cv2.matchTemplate(small_image, large_image, method)

    # We want the minimum squared difference
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx,MPy = min_loc

    # Step 2: Get the size of the template. This is the same size as the match.
    trows, tcols = small_image.shape[:2]

    # Step 3: Draw the rectangle on large_image
    cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)
    return large_image,max_val


def sift_image(rate,img1, img2):

    # Initiate SIFT detector
    # sift = cv2.SIFT()

    sift = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test

    good = []
    for m, n in matches:
        if m.distance < rate * n.distance:
            good.append([m])


    # cv2.drawMatchesKnn expects list of lists as matches.

    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
    #
    # plt.imshow(img3),plt.show()
    # print(good)
    # print(len(good))

    return len(good)



def find_image_in_db(rate, small_img):
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    org_s_img = np.copy(gray)

    # kernel = np.ones((3, 3), np.uint8)
    # gray = cv2.dilate(gray, kernel, iterations=1)
    # gray = cv2.erode(gray, kernel, iterations=1)

    # # image sharpener
    #


    """
    good code
    """
    list_match = []
    list_index = []
    # pass over the db of all images
    # for c_img in os.listdir('DB'):
    images = [cv2.imread(file) for file in glob.glob("pictures/DB/*.png")]
    for i, large_img in enumerate(images):
        num_much = sift_image(rate, gray.astype(np.uint8), large_img.astype(np.uint8))
        list_match.append([num_much,i])

    list_match = sorted(list_match, key=lambda x:x[0])
    list_match = list_match[-30:]
    # list_match = reversed(list_match)
    # for i in list_match:
    #     plt.imshow(images[i[1]])
    #     plt.show()
    print(list_match)
    return [images[x[1]] for x in list_match]

def ex33(img2):
    max_p = 0
    parameters = []
    images = [cv2.imread(file) for file in glob.glob("pictures/DB/*.png")]


    for j, img1 in enumerate(images):
        if j ==260:
            plt.imshow(img1)
            plt.show()
        sift = cv2.ORB_create()
        bf = cv2.BFMatcher()
        kp2, des2 = sift.detectAndCompute(img2,None)
        index_params= dict(algorithm = 6,
                                 table_number = 6,
                                 key_size = 12,
                                 multi_probe_level = 1)

        search_params = dict(checks=500)   # or pass empty dictionary
        # Read the images from the file
        img1 = cv2.imread('pictures/DB/ ({}).png'.format(j))
        kp1, des1 = sift.detectAndCompute(img1,None)

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        size =0
        # ratio test as per Lowe's paper
        good = []
        # ratio test as per Lowe's paper
        for m in matches:
            if len(m) > 0 and m[0].distance < 0.7*m[-1].distance:
                good.append(m[0])
        # print(size)

        parameters.append([len(good), j])

        # draw_params = dict(matchColor = (0,255,0),
        #                    singlePointColor = (255,0,0),
        #                    matchesMask = matchesMask,
        #                    flags = 0)
        #
        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

        # plt.imshow(img3,),plt.show()
    parameters = sorted(parameters, key = lambda x:x[0])
    parameters = parameters[-100:]
    print(parameters)

            # cv2.imshow("Img1", img1)
            # cv2.imshow("Img2", img2)


def clean_frame(thresh):
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    return cv2.boundingRect(cnt)


def clean_pic(i, img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    x,y,w,h = clean_frame(thresh)
    crop = img[y:y+h,x:x+w]
    img = cv2.medianBlur(crop,3)
    if i == 3:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
    if i == 1:
        img = img[30:, 30:]
        kernel = np.ones((3,3),np.uint8)

        # plt.imshow(img), plt.show()
    elif i == 4 or i == 2:
        mask = img.copy()
        _,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        img = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # img = cv2.filter2D(img, -1, kernel)
        # img = cv2.medianBlur(img,5)

    return img

def check_match(small_image, images):
    pic, rate = None, 0
    for i, large_image in enumerate(images):
        # Read the images from the file
        if small_image.shape[1] > large_image.shape[1]:
            small_image = small_image[0:small_image.shape[0], 0:large_image.shape[1]]
        found_img, rate_val = find_the_right_pic(small_image,large_image)
        # plt.imshow(large_image)
        # plt.title(i)
        # plt.show()
        print("{} - {}".format(i,rate_val))
        if rate_val > rate:
            pic = found_img
            rate = rate_val

    return pic

def check_for_circle(src):
     # Loads an image
    kernel = np.ones((3,3),np.uint8)
    src = cv2.erode(src, kernel, iterations=1)
    src = cv2.dilate(src, kernel, iterations=1)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        return -1


    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=45, param2=60 ,minRadius=6, maxRadius=50)
    if circles is None:
        return False
    return True

def clean_db(images):
    return [img for img in images if check_for_circle(img) == False]


def get_rate(i):
    if i == 0:
        return 0.75#v no blur
    elif i == 1:
        return 0.3
    elif i == 2:
        return 0.6#v no blur
    elif i == 3:
        return 0.6 #v blure
    elif i == 4:
        return 0.9


def activate_ex3():
    small_images = []
    large_images = []
    images = [cv2.imread(file) for file in glob.glob("pictures/q3/*.png")]
    for i, small_img in enumerate(images):
        if i == 4:
            rate = get_rate(i)
            small_img = clean_pic(i, small_img.copy())

            _, temp = cv2.threshold(small_img,155,255,cv2.THRESH_BINARY)# 4 works for 155

            img_list = find_image_in_db(rate, temp)
            if i == 4:
                # small_img = small_img[125:, 70:]
                plt.imshow(small_img), plt.show()
                pic = check_match(small_img, img_list)
                # plt.imshow(small_img), plt.show()
                if pic is not None:
                    plt.imshow(pic), plt.show()

    # small_images.a`   1ppend(path)
    # large_images.append(large_img)
    # cv2.imshow("large", large_img) nb  bv c dg
    # cv2.waitKey()



def activate_ex4():
    pass


if __name__ == '__main__':
    # activate_ex1()
    # activate_ex2()
    activate_ex3()
    # activate_ex4()




