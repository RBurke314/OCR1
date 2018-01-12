import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test4.jpg')

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(211),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test4.jpg')

blur = cv2.blur(img,(5,5))

plt.subplot(211),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


blurg = cv2.GaussianBlur(img,(5,5),0)
plt.subplot(211),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(blurg),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

median = cv2.medianBlur(img, 5)
plt.subplot(211),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(median),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

blurb = cv2.bilateralFilter(img,9,75,75)
plt.subplot(211),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(blurb),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()