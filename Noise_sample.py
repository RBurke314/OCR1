import cv2
import numpy as np

# To display a single image in a window
# Window is destroyed on pressing any key
def display(windowName, image):
  cv2.namedWindow(windowName, 1)
  cv2.imshow(windowName, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# Read image
img = cv2.imread('test4.jpg')
# Convert to grayscale image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
display('gray', gray)
# Convert to binary image
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
display('binary', thresh)

# noise removal
# to remove any small white noises use morphological opening
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
sure_bg = cv2.dilate(opening,kernel,iterations=3)
display('Sure Background', sure_bg)

#dist_transform = cv2.distanceTransform(opening,cv2.CV_DIST_L2,5)
dist_transform = cv2.distanceTransform(opening,cv.cv2.CV_DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
display('Sure Foreground', sure_fg)

# Finding unknown region
unknown = cv2.subtract(sure_bg,sure_fg)
display('unknown area', unknown)