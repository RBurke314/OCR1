#!/usr/bin/python
# Version 1
import cv2
import OCR_FUNCTION
import os
import pyautogui

Opnames = ['Resize', 'GrayScale',  'Adjust', 'Normalise', 'Threshold', 'Blur', 'Morph']
subOps1 = ['Gamma', 'Equalise 1', 'Equalise 2']
subOps2 = []
subOps3 = []
subOps4 = []
subOps5 = []
names = ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty']
image = 'test4.jpg'

def operationMenu ():
    pyautogui.hotkey('Ctrl', ';')
    print (50 * '-')
    print ("        Image Processing Operations")
    print (50 * '-')
    print ("1. %s" % (Opnames[0])),
    print ("2. %s" % (Opnames[1])),
    print ("3. %s" % (Opnames[2])),
    print ("4. %s" % (Opnames[3])),
    print ("5. %s" % (Opnames[4])),
    print ("6. %s" % (Opnames[5])),
    print ("7. %s" % (Opnames[6]))
    print (90 * '-')
    return
def menu1():
    pyautogui.hotkey('Ctrl', ';')
    print (50 * '-')
    print ("        Resize - Var1 = Scaling Factor e.g 2")
    print (50 * '-')
    return
def menu2():
    pyautogui.hotkey('Ctrl', ';')
    print (50 * '-')
    print ("        Grayscale")
    print (50 * '-')
    return
def menu3():
    pyautogui.hotkey('Ctrl', ';')
    print (50 * '-')
    print ("        Adjust")
    print (50 * '-')
    submenu1()
    return
def menu4():
    return
def menu5():
    return
def menu6():
    return
def menu7():
    return
def submenu1():
    print ("1. %s" % (Opnames[0])),
    print ("2. %s" % (Opnames[1])),
    print ("3. %s" % (Opnames[2])),
    print ("4. %s" % (Opnames[3])),
    print ("5. %s" % (Opnames[4])),
    print ("6. %s" % (Opnames[5])),
    print ("7. %s" % (Opnames[6]))
    print (90 * '-')
    return



Join = raw_input('Choose New Image [Y or N]\n')
if Join.lower() == 'yes' or Join.lower() == 'y':
    image = raw_input('Type Image Name: ')
elif Join.lower() == 'no' or Join.lower() == 'n':
    image = image
else:
    print ("No Answer Given")
original = OCR_FUNCTION.readImage(image)
OCR_FUNCTION.showImage('Original', original)

while True:
    k = cv2.waitKey(1) & 0xFF
    # press 'q' to exit
    if k == ord('q'):
        break
    ## Show menu ##
    print (50 * '-')
    print ("        Image Processing Pipeline Editor")
    print (50 * '-')
    print ("1. %s"% (names[0])),
    print ("2. %s"% (names[1])),
    print ("3. %s"% (names[2])),
    print ("4. %s"% (names[3])),
    print ("5. %s"% (names[4])),
    print ("6. %s"% (names[5])),
    print ("7. %s"% (names[6])),
    print ("8. %s"% (names[7])),
    print ("9. %s"% (names[8])),
    print ("10. %s"% (names[9]))

    print (90 * '-')




    ## Get input ###
    choice = raw_input('Choose Pipeline Element [1-%s] : '%len(names))
    ### Convert string to int type ##
    choice = int(choice)

    ### Take action as per selected menu-option ###
    if choice == 1:
        operationMenu()
        choice1 = raw_input('Choose Operation [1-%s] : '%len(Opnames))
        choice1 = int(choice1)
        if choice1 == 1:
            names[choice - 1] = Opnames[choice1 - 1]
            menu1()
            val_1 = raw_input('Var1 = \n')
            Join = raw_input('Are you sure ?? [Y or N]\n')
            if Join.lower() == 'yes' or Join.lower() == 'y':
                image1 = OCR_FUNCTION.resizeImage(original, 2)
            elif Join.lower() == 'no' or Join.lower() == 'n':
                image1 = image
            else:
                print ("No Answer Given")


        if choice1 == 2:
            names[choice - 1] = Opnames[choice1 - 1]
            menu2()
            Join = raw_input('Are you sure ?? [Y or N]\n')
            if Join.lower() == 'yes' or Join.lower() == 'y':
                image1 = OCR_FUNCTION.grayscale(original)
            elif Join.lower() == 'no' or Join.lower() == 'n':
                image1 = image
            else:
                print ("No Answer Given")
        if choice1 == 3:
            names[choice - 1] = Opnames[choice1 - 1]
            menu3()
            Join = raw_input('Are you sure ?? [Y or N]\n')
            if Join.lower() == 'yes' or Join.lower() == 'y':
                image1 = OCR_FUNCTION.grayscale(original)
            elif Join.lower() == 'no' or Join.lower() == 'n':
                image1 = image
            else:
                print ("No Answer Given")


    elif choice == 2:
        pyautogui.hotkey('Ctrl', ';')
        operationMenu()
        choice1 = raw_input('Choose Operation [1-%s] : '%len(Opnames))
        choice = int(choice)
    elif choice == 3:
        pyautogui.hotkey('Ctrl', ';')
        operationMenu()
        choice1 = raw_input('Choose Operation [1-%s] : '%len(Opnames))
        choice = int(choice)
    else:  ## default ##
        pyautogui.hotkey('Ctrl', ';')
        operationMenu()
        choice1 = raw_input('Choose Operation [1-%s] : '%len(Opnames))
        choice = int(choice)



