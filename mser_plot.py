#-*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt


img = cv2.imread("Samples//3.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mser = cv2.MSER(_delta = 1)
regions = mser.detect(gray, None)
 
# print len(regions) # 21개

# plot original image 
plt.subplot(7, 7, 48), plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
 
# plot edge map
edges = cv2.Canny(img,100,200)
plt.subplot(7, 7, 49), plt.imshow(edges, cmap = 'gray')
plt.title('Edge Map'), plt.xticks([]), plt.yticks([])
 
for i, region in enumerate(regions):
    clone = img.copy()
    cv2.drawContours(clone, region.reshape(-1,1,2), -1, (0, 255, 0), 1)
    (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
    cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 0, 0), 1)
    plt.subplot(7, 7, i+1), plt.imshow(clone)
    plt.title('Contours'), plt.xticks([]), plt.yticks([])

plt.show()
        
        

