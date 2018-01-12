
import cv2
import numpy as np

#source = (r'C:/Users/Rob/dev/VisionSystems/OCR/OCR1/DataBase/test/test_snippet.png')
source = r'3076.png'

img = cv2.imread(source)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ESC = 27

# create a black image with size 200x200 (in grayscale)
#img = np.zeros((200, 200), dtype=np.uint8)
# set the center of image to be a 50x50 white rectangle
#img[50:150, 50:150] = 255

# threshold the image
# if any pixels that have value higher than 127, assign it to 255
ret, threshed_img = cv2.threshold(img, 127, 255, 0)


# find contour in image
# cv2.RETR_TREE retrieves the entire hierarchy of contours in image
# if you only want to retrieve the most external contour
# use cv.RETR_EXTERNAL
image, contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)
# convert image back to BGR
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# draw contours onto image
img = cv2.drawContours(color_img, contours, -1, (0, 255, 0), 2)

cv2.imshow("contours", img)
cv2.waitKey(0)



cv2.destroyAllWindows


source = (r'C:/Users/Rob/dev/VisionSystems/OCR/OCR1/DataBase/test/test_snippet.png')


# read and scale down image
#img = cv2.pyrDown(cv2.imread(source, cv2.IMREAD_UNCHANGED))
img = cv2.imread(source, cv2.IMREAD_UNCHANGED)
gaussian_3 = cv2.GaussianBlur(img, (9, 9), 10.0)
unsharp_image = cv2.addWeighted(img, 1.5, gaussian_3, -0.1, 0, image)
cv2.imshow("sharp", unsharp_image)
cv2.waitKey(0)
# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(unsharp_image, cv2.COLOR_BGR2GRAY),
                                  127, 255, cv2.THRESH_BINARY)
# find contours and get the external one
image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

cv2.imshow("contours", img)
cv2.waitKey(0)

ESC = 27

cv2.destroyAllWindows()