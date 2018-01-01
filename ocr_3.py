

from PIL import Image
import numpy as np
import pytesseract
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = 'C://tesseract//tesseract.exe'

tessdata_dir_config = '--tessdata-dir "C://tesseract//tessdata"'

"""newjpgtxt = open(txtapp,"rb").read()
g= open("out.jpg","w")
g.write(base64.decodestring(newjpgtxt))
g.close()
filename=r'out.jpg'"""

filename=r'test4.jpg'

image=cv2.imread(filename)
cv2.imshow('Original',image)
cv2.waitKey(0)

#////////////////////////////////////////////////
img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray',img)
cv2.waitKey(0)

_,gray = cv2.threshold(img,145,255,cv2.THRESH_BINARY)

#gray= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, 1)
cv2.imshow('Thres',gray)
cv2.waitKey(0)
# Apply dilation and erosion to remove some noise
kernel = np.ones((1, 1), np.uint8)
img = cv2.dilate(gray, kernel, iterations=2)
cv2.imshow('Dilate 1',img)
cv2.waitKey(0)
img = cv2.erode(img, kernel, iterations=5)
cv2.imshow('Erode',img)
cv2.waitKey(0)
img = cv2.dilate(img, kernel, iterations=2)
cv2.imshow('Dilate 2',img)
cv2.waitKey(0)

img=cv2.medianBlur(img,1
                   )
cv2.imshow('blur',img)
cv2.waitKey(0)

filename2='script_img2.jpg'
cv2.imwrite(filename2,img)
img = Image.open(filename2)

#print(pytesseract.image_to_string(Image.open('C:/Users/Rob/dev/VisionSystems/OCR/test1.png'),lang='eng', config = tessdata_dir_config))
print(pytesseract.image_to_string(img,lang='eng', config = tessdata_dir_config))

# Apply dilation and erosion to remove some noise
#kernel = np.ones((1, 1), np.uint8)
#img = cv2.dilate(img, kernel, iterations=1)
#img = cv2.erode(img, kernel, iterations=5)


#img=cv2.medianBlur(img,3 )

#cv2.imwrite(filename + "removed_noise.png", img)
"""
#  Apply threshold to get image with only black and white
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
cv2.imshow('Adaptive Threshold',img)
cv2.waitKey(0)
"""


#gray=cv2.threshold(img,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

#img=cv2.medianBlur(img,3 )