

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
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
#////////////////////////////////////////////////
#Resize
dst = cv2.resize(image, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
#Grayscale
gray=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
#Normalise
equ = cv2.equalizeHist(gray)


def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

gamma = 2.5                                  # change the value here to get different result
adjusted = adjust_gamma(gray, gamma=gamma)

float_gray = gray.astype(np.float32) / 255.0
blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
num = float_gray - blur

blur = cv2.GaussianBlur(num*num, (0, 0), sigmaX=20, sigmaY=20)
den = cv2.pow(blur, 0.5)
gray = num / den
cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

hist,bins = np.histogram(gray2.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max() #/////

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))#/////Could be better than equalise Hist CHeCK!!!
cl1 = clahe.apply(gray2)

#Unsharp
gaussian_3 = cv2.GaussianBlur(equ, (9,9), 10.0)
unsharp_image = cv2.addWeighted(equ, 1.5, gaussian_3, -0.5, 0, equ)

img = cv2.medianBlur(unsharp_image,1)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(th2,-1,kernel)
blur = cv2.blur(th2,(5,5))

blurg = cv2.GaussianBlur(th2,(5,5),0)
median = cv2.medianBlur(th2, 5)
blurb = cv2.bilateralFilter(th2,9,75,75)



font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(gray,'GrayScaled',(5,15), font, 0.5,(0,0,0),2,cv2.LINE_AA)
#cv2.putText(equ,'Normalized',(5,15), font, 0.5,(0,0,0),2,cv2.LINE_AA)
#cv2.putText(unsharp_image,'Unsharp',(5,15), font, 0.5,(0,0,0),2,cv2.LINE_AA)
res = np.vstack((gray,equ,gaussian_3,unsharp_image)) #stacking images side-by-side
cv2.imshow('Norm',res)
cv2.waitKey(0)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

plt.subplot(211),plt.imshow(th2),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()


plt.subplot(211),plt.imshow(th2),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


plt.subplot(211),plt.imshow(th2),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(blurg),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


plt.subplot(211),plt.imshow(th2),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(median),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


plt.subplot(211),plt.imshow(th2),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(blurb),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

#_,gray = cv2.threshold(equ,145,255,cv2.THRESH_BINARY)
cv2.imshow('Thres',gray)
cv2.waitKey(0)
#gray= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, 1)

# Apply dilation and erosion to remove some noise
kernel = np.ones((1, 1), np.uint8)
img = cv2.erode(gray, kernel, iterations=5)
cv2.imshow('Erode',img)
cv2.waitKey(0)
img = cv2.dilate(img, kernel, iterations=2)
cv2.imshow('Dilate',img)
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