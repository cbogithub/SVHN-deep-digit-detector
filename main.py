#-*- coding: utf-8 -*-

import cv2

img = cv2.imread("Samples//1.png")
gaussian_3 = cv2.GaussianBlur(img, (9,9), 0)
cv2.addWeighted(img, 7.5, gaussian_3, -6.5, 0, img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mser = cv2.MSER(_delta = 1)
regions = mser.detect(gray, None)

print len(regions)

# loop over the contours
for region in regions:
    # fit a bounding box to the contour
    clone = img.copy()
    (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imshow('img', clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

