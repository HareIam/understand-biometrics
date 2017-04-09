import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture('traffic.mp4')
VideoW=int(cap.get(cv2 .CAP_PROP_FRAME_WIDTH))
VideoH=int(cap.get(cv2 .CAP_PROP_FRAME_HEIGHT))
VideoFPS=int(cap.get(cv2 .CAP_PROP_FPS))
frameCount=int(cap.get(cv2 .CAP_PROP_FRAME_COUNT))


_, img = cap . read ()
avgImg = np. float32 (img)
for fr in range (1 , frameCount ):
    _, img = cap . read ()
    avgImg1 = np.float32(img)
    if _ == True:
        avgImg=avgImg+avgImg1

avgImg= 1/(frameCount)*avgImg

avgImg1=np.uint8(avgImg)
cv2.imwrite('average.jpg',avgImg1)
cv2.imshow('testd', avgImg1)
cv2.waitKey(0)
cap . release()
