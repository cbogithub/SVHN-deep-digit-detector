#-*- coding: utf-8 -*-

import cv2

img = cv2.imread("Samples//1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clone = img.copy()
mser = cv2.MSER()
regions = mser.detect(gray, None)

# loop over the contours
for region in regions:
    # fit a bounding box to the contour
    (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', clone)
cv2.waitKey(0)
cv2.destroyAllWindows()

