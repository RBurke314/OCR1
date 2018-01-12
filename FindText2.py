import cv2


def captch_ex(file_name):
    img = cv2.imread(file_name)

    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', img2gray)
    cv2.waitKey(0)
    img2gray = cv2.medianBlur(img2gray, 1)
    cv2.imshow('Blur', img2gray)
    cv2.waitKey(0)

    #ret, mask = cv2.threshold(img2gray, 185, 255, cv2.THRESH_BINARY)
    #mask = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 30)
    ret, mask = cv2.threshold(img2gray, 127, 255, cv2.THRESH_TOZERO)


    #jhg = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
    cv2.imshow('Threshold', mask)
    cv2.waitKey(0)
    #image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    #cv2.imshow('And', image_final)
    #cv2.waitKey(0)
    ret, new_img = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV
    _,fds = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV
    cv2.imshow('Thresh 2', fds)
    cv2.waitKey(0)
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,
                                                         2))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=1)  # dilate , more the iteration more the dilation
    cv2.imshow('dilated', dilated)
    cv2.waitKey(0)
    cv2.imwrite(file_name + "_Mask2.jpg", dilated)
    # for cv2.x.x

    #contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours
    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours"
    # for cv3.x.x comment above line and uncomment line below

    #image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk
    cv2.imshow('captcha_result', img)
    cv2.waitKey()


file_name = 'img_set_1.jpg'

captch_ex(file_name)