import cv2
import numpy as np

def nothing(x):
    pass

#window_title = '%s - %s' % (thumbnail_basename, title)
# Create a black image, a window
#img = r'test3.jpg'
#img = r'FOR OCR.jpg'
#img = r'test4.jpg'
img = r'test4.jpg'
cv2.namedWindow('image')
cv2.resizeWindow('image', 600,600)
cv2.namedWindow('result')

TrackbarName0 = 'Gray and Thresh'
TrackbarName1 = 'Morph'
TrackbarName2 = 'erode'
TrackbarName3 = 'It'
TrackbarName4 = ''

TrackbarSave = 'Save?'
# create trackbars for color change
# create switch for ON/OFF functionality
#switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(TrackbarName0, 'image',0,1,nothing)
cv2.createTrackbar(TrackbarName1,'image',1,10,nothing)
cv2.createTrackbar(TrackbarName2,'image',1,10,nothing)
cv2.createTrackbar(TrackbarName3,'image',0,10,nothing)
cv2.createTrackbar(TrackbarName4,'image',0,10,nothing)
cv2.createTrackbar(TrackbarSave, 'result',0,1,nothing)



while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        print txt_result
        break

    image = cv2.imread(img)
    PipeImg = image


    # get current positions of four trackbars
    TrackbarPos0 = cv2.getTrackbarPos(TrackbarName0, 'image')
    if TrackbarPos0 == 0:
        PipeImg = image
    else:
        PipeImg = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        PipeImg = cv2.adaptiveThreshold(PipeImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2)

    TrackbarPos1 = cv2.getTrackbarPos(TrackbarName1, 'image')

    #Holder1 = np.zeros(Holder0.shape, np.uint8)
    #cv2.threshold(PipeImg, TrackbarPos1, 255, cv2.THRESH_BINARY, PipeImg)
    kernel = np.ones((TrackbarPos1, TrackbarPos1), np.uint8)
    #PipeImg = cv2.dilate(PipeImg, kernel, iterations=TrackbarPos2)
    PipeImg = cv2.morphologyEx(PipeImg, cv2.MORPH_CLOSE, kernel)

    TrackbarPos2 = cv2.getTrackbarPos(TrackbarName2, 'image')
    TrackbarPos3 = cv2.getTrackbarPos(TrackbarName3, 'image')

    # Holder1 = np.zeros(Holder0.shape, np.uint8)
    # cv2.threshold(PipeImg, TrackbarPos1, 255, cv2.THRESH_BINARY, PipeImg)
    kernel = np.ones((TrackbarPos2, TrackbarPos2), np.uint8)
    PipeImg = cv2.erode(PipeImg, kernel, iterations=TrackbarPos3)

    PipeImg = cv2.medianBlur(PipeImg, 1)

    #Operation


    # Operation

    TrackbarPos4 = cv2.getTrackbarPos(TrackbarName4, 'image')
    # Operation


    cv2.imshow('result', PipeImg)

    TrackbarSavePos = cv2.getTrackbarPos(TrackbarSave, 'result')
    if TrackbarSavePos == 0:
        txt_result = ('Not Saved')
    else:
        filename = 'For OCR.jpg'
        cv2.imwrite(filename, PipeImg)

        txt_result = (TrackbarName0, TrackbarPos0,
                      TrackbarName1, TrackbarPos1,
                      TrackbarName2, TrackbarPos2,
                      TrackbarName3, TrackbarPos3,
                      TrackbarName4, TrackbarPos4,

                  )




cv2.destroyAllWindows()