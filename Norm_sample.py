import cv2
import numpy as np

img = cv2.imread('test4.jpg')
cv2.imshow('origin', img)
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

float_gray = gray.astype(np.float32) / 255.0

blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
num = float_gray - blur

blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=20, sigmaY=20)
den = cv2.pow(blur, 0.5)

gray = num / den

cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

cv2.imwrite("./debug.png", gray * 255)
cv2.imshow('debug', gray)
cv2.waitKey(0)