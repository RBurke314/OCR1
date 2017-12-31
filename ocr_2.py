

from PIL import Image
import numpy as np
import pytesseract
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = 'C://tesseract//tesseract.exe'

tessdata_dir_config = '--tessdata-dir "C://tesseract//tessdata"'

newjpgtxt = open(txtapp,"rb").read()
g= open("out.jpg","w")
g.write(base64.decodestring(newjpgtxt))
g.close()
#


filename=r'out.jpg'

image=cv2.imread(filename)
#cv2.imshow('Original',image)
#cv2.waitKey(0)

img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('GrayScaled',img)
#cv2.waitKey(0)

# Apply dilation and erosion to remove some noise
kernel = np.ones((1, 1), np.uint8)
#img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=5)
#cv2.imshow('Eroded',img)
#cv2.waitKey(0)

img=cv2.medianBlur(img,3 )
#cv2.imshow('Blurred',img)
#cv2.waitKey(0)
# Write image after removed noise
#cv2.imwrite(filename + "removed_noise.png", img)
"""
#  Apply threshold to get image with only black and white
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
cv2.imshow('Adaptive Threshold',img)
cv2.waitKey(0)
# Write the image after apply opencv to do some ...
cv2.imwrite(filename + "thres.png", img)
"""

gray= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, 1)

#gray=cv2.threshold(img,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
#cv2.imshow('Thresholded',gray)
#cv2.waitKey(0)

# Apply dilation and erosion to remove some noise
kernel = np.ones((1, 1), np.uint8)
#img = cv2.dilate(img, kernel, iterations=1)
img = cv2.erode(gray, kernel, iterations=5)
#cv2.imshow('Eroded',img)
#cv2.waitKey(0)

img=cv2.medianBlur(img,3 )
#cv2.imshow('Blurred',img)
#cv2.waitKey(0)


filename='script_img.png'
cv2.imwrite(filename,gray)



#img = Image.open('C:/Users/Rob/dev/VisionSystems/OCR/test1.png')

img = Image.open(filename)


#print(pytesseract.image_to_string(Image.open('C:/Users/Rob/dev/VisionSystems/OCR/test1.png'),lang='eng', config = tessdata_dir_config))

print(pytesseract.image_to_string(img,lang='eng', config = tessdata_dir_config))