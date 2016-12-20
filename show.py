#-*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot_regions(img):
    # img 의 mesr region 과 edge map 을 plot 하는 함수.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mser = cv2.MSER(_delta = 1)
    regions = mser.detect(gray, None)
     
    n_regions = len(regions)
    n_rows = int(np.sqrt(n_regions)) + 1
    n_cols = int(np.sqrt(n_regions)) + 2
    
    # plot original image 
    plt.subplot(n_rows, n_cols, n_rows * n_cols-1)
    plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
      
    # plot edge map
    edges = cv2.Canny(img,10,20)
    plt.subplot(n_rows, n_cols, n_rows * n_cols)
    plt.imshow(edges, cmap = 'gray')
    plt.title('Edge Map'), plt.xticks([]), plt.yticks([])
      
    for i, region in enumerate(regions):
        clone = img.copy()
        cv2.drawContours(clone, region.reshape(-1,1,2), -1, (0, 255, 0), 1)
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
        cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 0, 0), 1)
        plt.subplot(n_rows, n_cols, i+1), plt.imshow(clone)
        plt.title('Contours'), plt.xticks([]), plt.yticks([])
     
    plt.show()
        
img = cv2.imread("Samples//3.png")
plot_regions(img)
