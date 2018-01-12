#!/usr/bin/env python


"""

"""

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
import cv2

#Global Variables
pytesseract.pytesseract.tesseract_cmd = 'C://tesseract//tesseract.exe'
tessdata_dir_config = '--tessdata-dir "C://tesseract//tessdata"'

source=r'test4.jpg'

interpol = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

#Functions
def printString(string):
    print string
def readImage(filename):
    image = cv2.imread(filename)
    return image
def showImage(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    return

def showStack(images):
    res = np.vstack((images))  # stacking images side-by-side
    return res
def resizeImage(image, factor, type): #2, inter_cubic
    resized = cv2.resize(image, None, fx=factor, fy=factor, interpolation= type)
    return resized
def grayscale(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray
def equalise(image):
    equ = cv2.equalizeHist(image)
    return equ
def equalise_2(image, clip, grid):# 2.0, (8,8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))  # /////Could be better than equalise Hist CHeCK!!!
    cl1 = clahe.apply(image)
    return cl1
def adjust_gamma(image, gamma=1.0):#gamma=1.0
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)
def normalise(image, alpha, beta):#0.0,1.0
    float_gray = image.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
    num = float_gray - blur
    blur = cv2.GaussianBlur(num * num, (0, 0), sigmaX=20, sigmaY=20)
    den = cv2.pow(blur, 0.5)
    gray = num / den
    norm = cv2.normalize(gray, dst=gray, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
    return norm
def normalise_2(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()  # /////
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[image]
    return img2
def unsharp(image, alpha, beta, gamma):#1.5, -0.5, 0
    gaussian_3 = cv2.GaussianBlur(equ, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(image, alpha, gaussian_3, beta, gamma, image)
    return unsharp_image
def thresh_binary(image, thresh): #blur before thresholding? #127
    ret, th1 = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return th1
def thresh_mean(image, size, c):#11,2 #3,5,7 , -5 to 5+
    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, size, c)
    return th2
def thresh_gaussian(image, size, c):
    th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, size, c)
    return th3
def blur(image, size):#5, 5
    blur = cv2.blur(image, (size, size))
    return blur
def blur_2d(image, size):# 5, 5
    kernel = np.ones((size, size), np.float32) / 25
    dst = cv2.filter2D(image, -1, kernel)
    return dst
def blur_median(image, size):#5
    median = cv2.medianBlur(image, size)
    return median
def blur_gaussian(image, size):
    blurg = cv2.GaussianBlur(image, (size, size), 0)
    return blurg
def blur_bilateral(image, diam, sigma):# 9, 75, 75
    blurb = cv2.bilateralFilter(image, diam, sigma, sigma)
    return blurb
def erode(image, size):#1, 1
    kernel = np.ones((size, size), np.uint8)
    img = cv2.erode(image, kernel, iterations=5)
    return img
def dilate(image, size):#1, 1
    kernel = np.ones((size, size), np.uint8)
    img = cv2.dilate(image, kernel, iterations=5)
    return img
def tesseract(image):
    filename2='script_img2.jpg'
    cv2.imwrite(filename2,image)
    img = Image.open(filename2)
    # print(pytesseract.image_to_string(Image.open('C:/Users/Rob/dev/VisionSystems/OCR/test1.png'),lang='eng', config = tessdata_dir_config))
    print(pytesseract.image_to_string(img, lang='eng', config=tessdata_dir_config))
    return

if __name__ == '__main__':
    import sys
    src = readImage(source)
    printString("Worked")
    showImage("Original",src)
    resized = resizeImage(src, 2, interpol[3])
    showImage("Resized", resized)
    gray = grayscale(resized)
    showImage("Grayscaled", gray)
    equ = equalise(gray)
    showImage("equalise", equ)
    equ_2 = equalise_2(gray, 2.0, 8)
    showImage("equalise_2", equ_2)
    adjusted = adjust_gamma(gray, gamma=2.5)
    showImage("adjusted", adjusted)
    norm = normalise(gray, 0.0, 1.0)
    showImage("Normalise", norm)
    norm_2 = normalise_2(gray)
    showImage("Normalise_2", norm_2)
    unsharp = unsharp(gray, 1.5, -0.5, 0)
    showImage("Unsharp", unsharp)
    thresh = thresh_binary(gray, 127)
    showImage("thresh_binary", thresh)
    thresh2 = thresh_mean(gray, 11, 2)
    showImage("thresh_mean", thresh2)
    thresh3 = thresh_gaussian(gray, 11, 2)
    showImage("thresh_gaussian", thresh3)
    blur = blur(gray, 5)
    showImage("blur", blur)
    blur2 = blur_2d(gray, 5)
    showImage("blur_2d", blur2)
    blur3 = blur_median(gray, 5)
    showImage("blur_median", blur3)
    blur4 = blur_gaussian(gray, 5)
    showImage("blur_gaussian", blur4)
    blur5 = blur_bilateral(gray, 9, 75)
    showImage("blur_bilateral", blur5)
    erode = erode(gray, 1)
    showImage("erode", erode)
    dilate = dilate(gray, 1)
    showImage("dilate", dilate)
    tesseract(gray)
    showImage("Imagez", gray)


