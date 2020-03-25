from final_project import cv2, plt, glob, find_circles

path = "pictures/"
def activate_ex2():
    final = []
    for i in range(31,46):
        img = cv2.imread(path+"q2/000{}.png".format(i))
        pic = find_circles(img)
        final.append(pic)

    for i in range(0,len(final),3):
        plt.subplot(1,3,1), plt.imshow(final[i], 'gray')
        plt.xticks([]),plt.yticks([])

        plt.subplot(1,3,2), plt.imshow(final[i+1], 'gray')
        plt.xticks([]),plt.yticks([])

        plt.subplot(1,3,3), plt.imshow(final[i+2], 'gray')
        plt.xticks([]),plt.yticks([])

        plt.show()


if __name__ == '__main__':
    activate_ex2()
