import cv2
import numpy as np

# Q1
img = cv2.imread('bee.png')
cv2.namedWindow("bee1")
cv2.imshow("bee1", img)
cv2.waitKey(0)

# Q2
img = cv2.imread('bee.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("bee2", hsv)
cv2.imwrite("Q3-2.jpg",hsv)
cv2.waitKey(0)

# Q3
H, S, V = cv2.split(hsv)
im_Hist = cv2.equalizeHist(V)
cv2.imshow('bee3',im_Hist)
cv2.imwrite("Q3-3-1.jpg",im_Hist)
cv2.waitKey(0)
hist = cv2.calcHist([im_Hist],[0],None,[256],[0.0,255.0])
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
histImg = np.zeros([256, 256, 3], np.uint8)
hpt = int(0.9 * 256)
for h in range(256):
    intensity = int(hist[h] * hpt / maxVal)
    cv2.line(histImg, (h, 256), (h, 256 - intensity), [0, 0, 255])
cv2.imshow("bee4", histImg)
cv2.imwrite("Q3-3-2.jpg",histImg)
cv2.waitKey(0)

# Q4

img_histV= cv2.imread("Q3-3-1.jpg",1)
im_rgb = cv2.cvtColor(img_histV, cv2.COLOR_HSV2BGR)
cv2.imshow("bee5", im_rgb)
cv2.imwrite("Q3-4.jpg",im_rgb)
cv2.waitKey(0)