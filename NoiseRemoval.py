import numpy as np
import cv2
from matplotlib import pyplot as plt
#img = cv2.imread('test.png')
fact = 3
img = cv2.imread (r'C:/Users/Rob/dev/VisionSystems/OCR/OCR1/DataBase/test/test_snippet.png')

var1 = 10
var2 = 7
var3 = 21

#dst = cv2.fastNlMeansDenoisingColored(img,None,var1,var1,var2,var3)
#gaussian_3 = cv2.GaussianBlur(dst, (9, 9), 10.0)
#unsharp_image = cv2.addWeighted(dst, 1.5, gaussian_3, -0.7, 0, dst)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray, (1, 1))
#ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 33, 40)
kernel = np.ones((2, 2), np.uint8)
img = cv2.erode(th2, kernel, iterations=1)
img = (255-img)


mod1 = cv2.resize(gray, (0,0), fx=fact, fy=fact)
mod2 = cv2.resize(blur, (0,0), fx=fact, fy=fact)
mod3 = cv2.resize(th2, (0,0), fx=fact, fy=fact)
mod3 = cv2.resize(img, (0,0), fx=fact, fy=fact)
sum2 = [mod1,mod2,mod3]




cv2.imshow('one', mod3)

cv2.waitKey(0)