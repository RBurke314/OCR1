#!/usr/bin/env python


"""

"""

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
import cv2
import sys
import pyautogui

#Global Variables
pytesseract.pytesseract.tesseract_cmd = 'C://tesseract//tesseract.exe'
tessdata_dir_config = '--tessdata-dir "C://tesseract//tessdata"'
interpol = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

#Functions
def nothing(x):
    pass
def printString(string):
    print string
def readImage(filename):
    image = cv2.imread(filename)
    return image
def showImage(title, image):
    cv2.imshow(title, image)
    k = cv2.waitKey()
    if k == 32:
        pass
    elif k == 27:
        sys.exit()
    return
def showImage2(image):
    #cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.imshow('Frame', image)
    k = cv2.waitKey()
    if k == 32:
        pass
    elif k == 27:
        sys.exit()
    return
def imageStack(images):
    res = np.vstack((images))  # stacking images side-by-side
    return res
def resizeImage(image, factor, type): #2, inter_cubic
    resized = cv2.resize(image, None, fx=factor, fy=factor, interpolation= type)
    print 'Resized',
    return resized
def grayscale(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    print ', Grayscaled'
    return gray
def equalise(image):
    equ = cv2.equalizeHist(image)
    print 'Eqaulised 1',
    return equ
def equalise_2(image, clip, grid):# 2.0, (8,8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))  # /////Could be better than equalise Hist CHeCK!!!
    cl1 = clahe.apply(image)
    print ', Equalised 2',
    return cl1
def adjust_gamma(image, gamma=1.0):#gamma=1.0
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   print ', Gamma Adjusted'
   return cv2.LUT(image, table)
def normalise(image, alpha, beta):#0.0,1.0
    float_gray = image.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(float_gray, (0, 0), sigmaX=2, sigmaY=2)
    num = float_gray - blur
    blur = cv2.GaussianBlur(num * num, (0, 0), sigmaX=20, sigmaY=20)
    den = cv2.pow(blur, 0.5)
    gray = num / den
    norm = cv2.normalize(gray, dst=gray, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
    print 'Normalised',
    return norm
def normalise_2(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()  # /////
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[image]
    print ', Normalised 2',
    return img2
def unsharp(image, alpha, beta, gamma):#1.5, -0.5, 0
    gaussian_3 = cv2.GaussianBlur(image, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(image, alpha, gaussian_3, beta, gamma, image)
    print ', Unsharped'
    return unsharp_image
def thresh_binary(image, thresh): #blur before thresholding? #127
    ret, th1 = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    print 'Thresholded 1',
    return th1
def thresh_mean(image, size, c):#11,2 #3,5,7 , -5 to 5+
    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, size, c)
    print ', Thresholded 2',
    return th2
def thresh_gaussian(image, size, c):
    th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, size, c)
    print ', Thresholded 3'
    return th3
def blur(image, size):#5, 5
    blur = cv2.blur(image, (size, size))
    print 'Blur 1',
    return blur
def blur_2d(image, size):# 5, 5
    kernel = np.ones((size, size), np.float32) / 25
    dst = cv2.filter2D(image, -1, kernel)
    print ', Blur 2',
    return dst
def blur_median(image, size):#5
    median = cv2.medianBlur(image, size)
    print ', Blur 3',
    return median
def blur_gaussian(image, size):
    blurg = cv2.GaussianBlur(image, (size, size), 0)
    print ', Blur 4',
    return blurg
def blur_bilateral(image, diam, sigma):# 9, 75, 75
    blurb = cv2.bilateralFilter(image, diam, sigma, sigma)
    print ', Blur 5'
    return blurb
def erode(image, size):#1, 1
    kernel = np.ones((size, size), np.uint8)
    img = cv2.erode(image, kernel, iterations=5)
    print 'Eroded',
    return img
def dilate(image, size):#1, 1
    kernel = np.ones((size, size), np.uint8)
    img = cv2.dilate(image, kernel, iterations=5)
    print ', Dialated'
    return img
def tesseract(image):
    filename2='script_img2.jpg'
    cv2.imwrite(filename2,image)
    img = Image.open(filename2)
    # print(pytesseract.image_to_string(Image.open('C:/Users/Rob/dev/VisionSystems/OCR/test1.png'),lang='eng', config = tessdata_dir_config))
    print 'Tesseract Returned -',
    print(pytesseract.image_to_string(img, lang='eng', config=tessdata_dir_config))
    return

print 'Starting...'
Join = raw_input('Would you like to change input and scale??? [Y or N]\n')
if Join.lower() == 'yes' or Join.lower() == 'y':
    source = raw_input('Input Image: ')
    Join = raw_input('Enter Scaling Factor (e.g 2):\n')
    choice = int(Join)
    print 'Ok...'
elif Join.lower() == 'no' or Join.lower() == 'n':
    source = r'C:/Users/Rob/dev/VisionSystems/OCR/OCR1/DataBase/test/test_snippet.png'
    choice = int(2)
    print 'Ok...'
else:
    print ("No Answer Given")


pyautogui.hotkey('Ctrl', ';')
pyautogui.hotkey('Ctrl', ';')


print (50 * '-')
print ("            OCR Pipeline Manipulator")
print (50 * '-')
print 'Press Space to pass operation...'
print (30 * '-')
print 'Press Esc to Exit'
print (30 * '-')

if __name__ == '__main__':
    while True:
        src = readImage(source)#Read Image



        resized = resizeImage(src, choice, interpol[3])#Resize image
        showImage2(resized)
        #/////////////////////////////////////
        passed = resized

        #Attempt to clean image/ reduce noise before grayscale******************
        #Note that most algorythoms require grayscale
        #Thresholds usually need blur

        #Add track bar functionality to each window

        #/////Grayscale//////////////////////
        gray = grayscale(passed)
        showImage2(gray)
        #///////////////////////////
        passed = gray
        #/////Equalise//////////////////////
        img1 = equalise(passed)
        img2 = equalise_2(passed, 2.0, 8)
        img3 = adjust_gamma(passed, gamma=2.5)
        sum =[img1, img2, img3]
        stack = imageStack(sum)
        showImage('Adjust', stack)
        #//////////////////////////////////
        passed = passed
        #/////Normalise///////////////////////
        img1 = normalise(passed, 0.0, 1.0)
        img2 = normalise_2(passed)
        img3 = unsharp(passed, 1.5, -0.5, 0)
        sum = [img1, img2, img3]
        stack = imageStack(sum)
        showImage('Normalise', stack)
        #//////////////////////////////////////
        passed = passed
        #//////Threshold////////////////////////
        img1 = thresh_binary(passed, 127)
        img2 = thresh_mean(passed, 11, 2)
        img3 = thresh_gaussian(passed, 11, 2)
        sum = [img1, img2, img3]
        stack = imageStack(sum)
        showImage('Threshold', stack)
        #////////////////////////////////////
        passed = passed
        #/////Blur///////////////////////////////
        img1 = blur(passed, 5)
        img2 = blur_2d(passed, 5)
        img3 = blur_median(passed, 5)
        img4 = blur_gaussian(passed, 5)
        img5 = blur_bilateral(passed, 9, 75)
        sum = [img1, img2, img3, img4, img5]
        stack = imageStack(sum)
        showImage('Blur', stack)
        #/////////////////////////////////////
        passed = passed
        #/////Morph///////////////////////////
        img1 = erode(passed, 1)
        img2 = dilate(passed, 1)
        sum = [img1, img2]
        stack = imageStack(sum)
        showImage('Morph', stack)
        #///////////////////////////////////////
        passed = passed
        #/////////////////////////////////
        tesseract(passed)

        cv2.destroyAllWindows()


