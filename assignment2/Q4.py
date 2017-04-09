import cv2
import numpy as np
from scipy import signal

# Q2
img = cv2.imread('test.jpg', 0)
Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]

# Q3
Mx = signal.convolve2d(img, Gx, 'same')
My = signal.convolve2d(img, Gy, 'same')
M = np.sqrt(Mx**2 + My**2)
M = 255*M/np.amax(M)
M = np.array(M, dtype='uint8')

cv2.imshow("window", M)
cv2.imwrite("Q4.jpg",M)
cv2.waitKey(0)


