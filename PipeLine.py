

from PIL import Image
import numpy as np
import pytesseract
import cv2
import os
import sys
import pyautogui
pytesseract.pytesseract.tesseract_cmd = 'C://tesseract//tesseract.exe'

tessdata_dir_config = '--tessdata-dir "C://tesseract//tessdata"'


myList  = range(3, 1000)
myList2  = range(1, 30)
oddsList = [x for x in myList if x % 2 != 0]
oddsList2 = [x for x in myList2 if x % 2 != 0]
NewValue = (((70 - 0) * 100) / 100) + -50

source = r'C:/Users/Rob/dev/VisionSystems/OCR/OCR1/DataBase/test/test_snippet_3.png'

image = cv2.imread(source)
#cv2.imshow('Origin',image)
#cv2.waitKey(0)

resized = cv2.resize(image, None, fx=2, fy=2, interpolation= cv2.INTER_CUBIC)
gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray, (oddsList2[1], oddsList2[1]))
th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, oddsList[33], NewValue)
#cv2.imshow('Thresh',th2)
#cv2.waitKey(0)
filename2='script_img2.png'
cv2.imwrite(filename2,th2)
img = Image.open(filename2)
print(pytesseract.image_to_string(img, lang='eng', config=tessdata_dir_config))

