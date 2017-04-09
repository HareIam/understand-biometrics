
import numpy as np  

import cv2


# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS


cap = cv2.VideoCapture('traffic.mp4')
VideoW=int(cap.get(cv2 .CAP_PROP_FRAME_WIDTH))
VideoH=int(cap.get(cv2 .CAP_PROP_FRAME_HEIGHT))
VideoFPS=int(cap.get(cv2 .CAP_PROP_FPS))
frameCount=int(cap.get(cv2 .CAP_PROP_FRAME_COUNT))

success, img = cap.read ()
img2=np.float32(img)
framesave=[]
b1=[]
dataSet=[]
z=np.zeros((480,640,3))

#print(avgImg1)
for fr in range (0, frameCount):
    success,img = cap.read()
    avgImg = np.float32(img)
    framesave.append(avgImg)


for wid in range(0, VideoW):
    for Hig in range(0, VideoH):
        dataSet = []
        for fr in range(0, frameCount-1):
            dataSet.append(framesave[fr][Hig][wid].tolist())
            #print(dataSet)
        dataSet = np.float32(dataSet)
        compactnessb,labelsb,centersb = cv2.kmeans(dataSet,2,None,criteria,10,flags)
        count1 =0
        count0 =0
        for lab in range(0,frameCount-1):
            if labelsb[lab]==1:
               count1+=1
            else:
               count0+=1
        if count0>=count1:
            z[Hig,wid, : ]= centersb[0]
        else:
            z[Hig,wid, : ] = centersb[1]
merged=np.uint8(z)
cv2.imwrite('Q3.jpg',merged)
cv2.imshow('testd', merged)
cv2.waitKey(0)
cap.release()