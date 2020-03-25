from final_project import cv2, plt, glob, np
path = "pictures/"

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
    return large_image, max_val/min_val


def sift_image(bf, des2, des1):
    # BFMatcher with default params
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return len(good)



def find_image_in_db(bf,images_len,critical_points, des_small_img):
    max_match, index_img = 0, 0

    for i in range(images_len):
        # return the image with the maximum match
        matches = sift_image(bf,critical_points[i], des_small_img)
        if max_match < matches:
            max_match = matches
            index_img = i
    return index_img


def get_critical_points_DB(images, sift):
    critical_points = []
    for img in images:
        kp, des = sift.detectAndCompute(img,None)
        critical_points.append(des)
    return critical_points



def activate_ex3():
    images = [cv2.imread(file) for file in glob.glob(path+"DB/*.png")]
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()
    critical_points_db = get_critical_points_DB(images, sift)
    for i in range(1,6):
        small_img_org = cv2.imread(path +"q3/00031_{}.png".format(i))
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        small_img = cv2.filter2D(small_img_org, -1, kernel)
        kp, des_small_img = sift.detectAndCompute(small_img,None)
        #shows image with the recognized footprint
        org_pic = images[find_image_in_db(bf, len(images), critical_points_db, des_small_img)]
        plt.subplot(1, 2, 1), plt.imshow(small_img_org,'gray')
        plt.xticks([]),plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(org_pic,'gray')
        plt.xticks([]),plt.yticks([])
        plt.show()

if __name__ == '__main__':
    activate_ex3()
