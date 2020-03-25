from final_project import cv2, plt, np, find_circles
path = "pictures/"


def find_circles2(src):
     # Loads an image
    org = src.copy()
    _,src = cv2.threshold(src,109,255,cv2.THRESH_BINARY)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    # find circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=35, param2=35 ,minRadius=7, maxRadius=50)

    # draw the circles in the image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(org, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(org, center, 25, (255, 0, 255), 3)
    return org


def activate_ex4():
    pic1 = cv2.imread(path + 'q4/00009.JPG')
    pic2 = cv2.imread(path + 'q4/00108.JPG')
    org = pic1.copy()
    pic1_with_circles = find_circles2(org)
    pic2_with_circles = find_circles(pic2)
    # shows images with and without the marked circles that found
    plt.subplot(1, 4, 1), plt.imshow(pic1,'gray')
    plt.xticks([]),plt.yticks([])
    plt.subplot(1, 4, 2), plt.imshow(pic1_with_circles,'gray')
    plt.xticks([]),plt.yticks([])
    plt.subplot(1, 4, 3), plt.imshow(pic2,'gray')
    plt.xticks([]),plt.yticks([])
    plt.subplot(1, 4, 4), plt.imshow(pic2_with_circles,'gray')
    plt.xticks([]),plt.yticks([])
    plt.show()

if __name__ == '__main__':
    activate_ex4()
