import cv2

import numpy as np

Image=r'test3.jpg'
Image = cv2.imread(Image)
OperationNum = 10
TrackbarName=[None]*10
Holder = [None]*10
TrackbarPos = [None]*10
# Callback Function for Trackbar (but do not any work)
def nothing(x):
    pass

# Code here
def SimpleTrackbar(Image, WindowName):
 for i in range(OperationNum):
     # Generate trackbar Window Nam
     TrackbarName[i] = WindowName #+ "1"
     # Make Window and Trackbar
     cv2.namedWindow(WindowName)
     cv2.createTrackbar(TrackbarName[i], WindowName, 0, 255, nothing)
     # Allocate destination image
     Holder[i] = np.zeros(Image.shape, np.uint8)



 # Loop for get trackbar pos and process it
 while True:
  # Get position in trackbar
  TrackbarPos[0] = cv2.getTrackbarPos(TrackbarName[0], WindowName)
  # Apply threshold
  cv2.threshold(Image, TrackbarPos[0], 255, cv2.THRESH_BINARY, Holder[0])

  TrackbarPos[1] = cv2.getTrackbarPos(TrackbarName[1], WindowName)
  Holder[1] = cv2.cvtColor(Holder[0], cv2.COLOR_BGR2GRAY)
  # Show in window
  cv2.imshow(WindowName, Holder[2])




  # If you press "ESC", it will return value
  ch = cv2.waitKey(5)
  if ch == 27:
      print 'filter value?'
      break
 cv2.destroyAllWindows()
 return Holder

def main():
    SimpleTrackbar(Image, "Image")

if __name__ == "__main__":
    main()
