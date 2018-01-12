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
source = r'C:/Users/Rob/dev/VisionSystems/OCR/OCR1/DataBase/test/test_snippet_3.png'
#source = r'C:/Users/Rob/dev/VisionSystems/OCR/OCR1/DataBase/test/test_snippet_3.png'
#source = r'C:/Users/Rob/dev/VisionSystems/OCR/OCR1/DataBase/test/test_snippet_2.png'
#source = r'C:/Users/Rob/dev/VisionSystems/OCR/OCR1/blat.png'

#source = r'C:/Users/Rob/dev/VisionSystems/OCR/OCR1/img_set_2.png'

sfactor = 2
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
    elif k == 101:
        source = raw_input('Input Image: ')
        Join = raw_input('Enter Scaling Factor (e.g 2):\n')
        sfactor = int(Join)
        print 'Ok...'
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
def thresh_binary(image, thresh, state): #blur before thresholding? #127
    ret, th1 = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    if state == 0:
        print 'Thresholded 1',
    return th1
def thresh_mean(image, size, c, state):#11,2 #3,5,7 , -5 to 5+
    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, size, c)
    if state == 0:
        print ', Thresholded 2',
    return th2
def thresh_gaussian(image, size, c, state):
    th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, size, c)
    if state == 0:
        print ', Thresholded 3'
    return th3
def blur(image, size, state1):#5, 5
    blur = cv2.blur(image, (size, size))
    if state1 == 0:
        print 'Blur 1',
    return blur
def blur_2d(image, size, state1):# 5, 5
    kernel = np.ones((size, size), np.float32) / 25
    dst = cv2.filter2D(image, -1, kernel)
    if state1 == 0:
        print ', Blur 2',
    return dst
def blur_median(image, size, state1):#5
    median = cv2.medianBlur(image, size)
    if state1 == 0:
        print ', Blur 3',
    return median
def blur_gaussian(image, size, state1):
    blurg = cv2.GaussianBlur(image, (size, size), 0)
    if state1 == 0:
        print ', Blur 4',
    return blurg
def blur_bilateral(image, diam, sigma, state1):# 9, 75, 75
    blurb = cv2.bilateralFilter(image, diam, sigma, sigma)
    if state1 == 0:
        print ', Blur 5'
    return blurb
def erode(image, size, it, state2):#1, 1
    kernel = np.ones((size, size), np.uint8)
    img = cv2.erode(image, kernel, iterations=it)
    if state2 == 0:
        print 'Eroded',
    return img
def dilate(image, size, it, state2):#1, 1
    kernel = np.ones((size, size), np.uint8)
    img = cv2.dilate(image, kernel, iterations=it)
    if state2 == 0:
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
def threshold(image, state):
    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            break
        elif k == 27:
            sys.exit()
        TrackbarPos0 = cv2.getTrackbarPos(TrackbarName0, 'Threshold')
        TrackbarPos1 = cv2.getTrackbarPos(TrackbarName1, 'Threshold')
        TrackbarPos2 = cv2.getTrackbarPos(TrackbarName2, 'Threshold')
        NewValue = (((TrackbarPos2 - 0) * 100) / 100) + -50
        img1 = thresh_binary(image, TrackbarPos0, state)
        img2 = thresh_mean(image, oddsList[TrackbarPos1], NewValue, state)
        img3 = thresh_gaussian(image, oddsList[TrackbarPos1], NewValue, state)
        state = 1
        sum = [img1, img2, img3]
        stack = imageStack(sum)
        cv2.imshow('Threshold', stack)
    return img2

def deNoise(image, state3):
    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            break
        elif k == 27:
            sys.exit()
        if state3 == 0:
            print ', DeNoise'
        TrackbarPos6 = cv2.getTrackbarPos(TrackbarName6, 'deNoise')
        TrackbarPos7 = cv2.getTrackbarPos(TrackbarName7, 'deNoise')
        TrackbarPos8 = cv2.getTrackbarPos(TrackbarName8, 'deNoise')
        dst = cv2.fastNlMeansDenoisingColored(image, None, TrackbarPos6, TrackbarPos6, oddsList3[TrackbarPos7], oddsList3[TrackbarPos8])
        state3 = 1
        sum = [image, dst]
        stack = imageStack(sum)
        cv2.imshow('deNoise', stack)
    return dst
def blurImg(image, state1):
    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            break
        elif k == 27:
            sys.exit()
        TrackbarPos3 = cv2.getTrackbarPos(TrackbarName3, 'Blur')
        img1 = blur(image, oddsList2[TrackbarPos3], state1)
        img2 = blur_2d(image, oddsList2[TrackbarPos3], state1)
        img3 = blur_median(image, oddsList2[TrackbarPos3], state1)
        img4 = blur_gaussian(image, oddsList2[TrackbarPos3], state1)
        img5 = blur_bilateral(image, 9, 75, state1)
        state1 = 1
        sum = [img1, img2, img3, img4, img5]
        stack = imageStack(sum)
        #showImage('Blur', stack)
        cv2.imshow('Blur', stack)

    return img1
def morph(image, state2):
    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            break
        elif k == 27:
            sys.exit()
        TrackbarPos4 = cv2.getTrackbarPos(TrackbarName4, 'Morph')
        TrackbarPos5 = cv2.getTrackbarPos(TrackbarName5, 'Morph')
        img1 = erode(image, TrackbarPos4, TrackbarPos5, state2)
        img2 = dilate(image, TrackbarPos4, TrackbarPos5, state2)
        state2 = 1
        sum = [img1, img2]
        stack = imageStack(sum)
        cv2.imshow('Morph', stack)
    return image
print 'Starting...'


max_val = 1000
max_val2 = 150
myList  = range(3, max_val)
myList2  = range(1, 30)
myList3  = range(1, max_val2)
oddsList = [x for x in myList if x % 2 != 0]
oddsList2 = [x for x in myList2 if x % 2 != 0]
oddsList3 = [x for x in myList3 if x % 2 != 0]

pyautogui.hotkey('Ctrl', ';')


print (50 * '-')
print ("            OCR Pipeline Manipulator")
print (50 * '-')
print 'Press Space to pass operation...'
print (30 * '-')
print 'Press Esc to Exit'
print (30 * '-')
print 'Press E to change parameters'
print (30 * '-')
if __name__ == '__main__':
    while True:
        # Attempt to clean image/ reduce noise before grayscale******************
        # Note that most algorythoms require grayscale
        # Thresholds usually need blur

        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            break
        elif k == 101:
            if Join.lower() == 'yes' or Join.lower() == 'y':
                source = raw_input('Input Image: ')
                Join = raw_input('Enter Scaling Factor (e.g 2):\n')
                sfactor = int(Join)
                print 'Ok...'
            elif Join.lower() == 'no' or Join.lower() == 'n':
                source = source
                sfactor = int(2)
                print 'Ok...'
            else:
                print ("No Answer Given")
        elif k == 27:
            sys.exit()


        src = readImage(source)  # Read Image
        passed = src
        """# //////Denoise////////////////////////
        TrackbarName6 = 'Luminance'
        TrackbarName7 = 'Search Window'
        TrackbarName8 = 'Block size'
        cv2.namedWindow('deNoise')
        cv2.createTrackbar(TrackbarName6, 'deNoise', 10, 30, nothing)
        cv2.createTrackbar(TrackbarName7, 'deNoise', 7, 50, nothing)
        cv2.createTrackbar(TrackbarName8, 'deNoise', 21, 50, nothing)
        state3 = 0
        deNoiseImg = deNoise(passed, state3)
        # ////////////////////////////////////
        passed = deNoiseImg"""

        # ///////Resize//////////////////////////////
        resized = resizeImage(src, sfactor, interpol[3])#Resize image
        # /////////////////////////////////////
        passed = resized

        #/////Grayscale//////////////////////
        gray = grayscale(passed)
        graytoRGB = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        sum = [resized, graytoRGB]
        stack = imageStack(sum)
        showImage('Gray', stack)
        #///////////////////////////
        passed = gray
        # /////Blur///////////////////////////////
        TrackbarName3 = 'Blur'
        cv2.namedWindow('Blur')
        cv2.createTrackbar(TrackbarName3, 'Blur', 1, 10, nothing)
        state1 = 0
        blurimg = blurImg(passed, state1)
        # /////////////////////////////////////
        passed = blurimg

        #//////Threshold////////////////////////
        TrackbarName0 = 'Threshold Image 1'
        TrackbarName1 = 'Image 2 Size'
        TrackbarName2 = 'Image 2 Constant'
        cv2.namedWindow('Threshold')
        cv2.createTrackbar(TrackbarName0, 'Threshold', 127, 255, nothing)
        cv2.createTrackbar(TrackbarName1, 'Threshold', 33, 400, nothing)
        cv2.createTrackbar(TrackbarName2, 'Threshold', 70 , 100, nothing)
        state = 0
        threshImg = threshold(passed, state)
        #////////////////////////////////////
        passed = threshImg
        """# /////Normalise///////////////////////
        img1 = normalise(passed, 0.0, 1.0)
        img2 = normalise_2(passed)
        img3 = unsharp(passed, 1.5, -0.7, 0)
        sum = [img1, img2, img3]
        stack = imageStack(sum)
        showImage('Normalise', stack)
        # //////////////////////////////////////"""
        """passed = passed
        # /////Equalise//////////////////////
        img1 = equalise(passed)
        img2 = equalise_2(passed, 2.0, 8)
        img3 = adjust_gamma(passed, gamma=2.5)
        sum = [img1, img2, img3]
        stack = imageStack(sum)
        showImage('Adjust', stack)
        # //////////////////////////////////"""
        passed = threshImg
        """"#/////Blur///////////////////////////////
        TrackbarName3 = 'Blur'
        cv2.namedWindow('Blur')
        cv2.createTrackbar(TrackbarName3, 'Blur', 6, 10, nothing)
        state1 = 0
        blurimg = blurImg(passed, state1)
        #/////////////////////////////////////
        passed = passed"""
        #/////Morph///////////////////////////
        TrackbarName4 = 'Morph Size'
        TrackbarName5 = 'Morph It'
        cv2.namedWindow('Morph')
        cv2.createTrackbar(TrackbarName4, 'Morph', 2, 10, nothing)
        cv2.createTrackbar(TrackbarName5, 'Morph', 2 , 10, nothing)
        state2 = 0
        morphimg = morph(threshImg, state2)



        #///////////////////////////////////////
        passed = morphimg

        #/////////////////////////////////
        cv2.destroyAllWindows()
        tesseract(passed)




