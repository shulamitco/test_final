from final_project import hypot, cv2, np, plt
path = "pictures/"
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
    picture_path_4 = path+'q1_fromZira/4.JPG'
    picture_path_12 = path+'q1_fromZira/12.JPG'
    picture_path_13 = path+'q1_fromZira/13.JPG'
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

    for i, pic in enumerate(pictures):
        plt.subplot(1,3,i+1),plt.imshow(pic)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    titles2 = ["org", "pointed", "crop img"]
    pictures2 = [org2, img2, crop_img2]

    for i, pic in enumerate(pictures2):
        plt.subplot(1,3,i+1),plt.imshow(pic)
        plt.title(titles2[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    titles3 = ["org", "pointed", 'crop']
    pictures3 = [org3, img3, crop_img3]

    for i,pic in enumerate(pictures3):
        plt.subplot(1,3,i+1),plt.imshow(pic, 'gray')
        plt.title(titles3[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

if __name__ == '__main__':
    activate_ex1()
