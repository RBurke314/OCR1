import cv2
import numpy as np

def nothing(x):
    pass

#window_title = '%s - %s' % (thumbnail_basename, title)
# Create a black image, a window
img = r'test3.jpg'
cv2.namedWindow('image')
cv2.resizeWindow('image', 600,600)
cv2.namedWindow('result')

TrackbarName0 = 'Grayscale'
TrackbarName1 = 'Threshold'
TrackbarName2 = 'None'
TrackbarName3 = 'None'
TrackbarName4 = 'None'

TrackbarSave = 'Save?'
# create trackbars for color change
# create switch for ON/OFF functionality
#switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(TrackbarName0, 'image',0,1,nothing)
cv2.createTrackbar(TrackbarName1,'image',127,255,nothing)
cv2.createTrackbar(TrackbarName2,'image',127,255,nothing)
cv2.createTrackbar(TrackbarName3,'image',127,255,nothing)
cv2.createTrackbar(TrackbarName4,'image',127,255,nothing)
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

    TrackbarPos1 = cv2.getTrackbarPos(TrackbarName1, 'image')
    #Holder1 = np.zeros(Holder0.shape, np.uint8)
    cv2.threshold(PipeImg, TrackbarPos1, 255, cv2.THRESH_BINARY_INV, PipeImg)

    TrackbarPos2 = cv2.getTrackbarPos(TrackbarName2, 'image')
    #Operation

    TrackbarPos3 = cv2.getTrackbarPos(TrackbarName3, 'image')
    # Operation

    TrackbarPos4 = cv2.getTrackbarPos(TrackbarName4, 'image')
    # Operation


    cv2.imshow('result', PipeImg)

    TrackbarSavePos = cv2.getTrackbarPos(TrackbarSave, 'result')
    if TrackbarSavePos == 0:
        txt_result = ('Not Saved')
    else:

        txt_result = (TrackbarName0, TrackbarPos0,
                      TrackbarName1, TrackbarPos1,
                      TrackbarName2, TrackbarPos2,
                      TrackbarName3, TrackbarPos3,
                      TrackbarName4, TrackbarPos4,

                  )




cv2.destroyAllWindows()