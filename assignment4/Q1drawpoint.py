import numpy as np
from math import *
import matplotlib.pyplot as plt

def TransT1(a):
    T1 = [[1, 0, -a[0]], [0, 1, -a[1]], [0, 0, 1]]
    return T1


def RotateR(theta):
    R = [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]
    return R


def TransT2(C):
    T1 = [[1, 0, C[0]], [0, 1, C[1]], [0, 0, 1]]
    return T1


def TransP(T2, R, T1, P):
    P1 = np.dot(T2, np.dot(R, np.dot(T1, P)))
    return P1

P=[[3],[3],[1]]
C=[2,2]
plt.plot(3, 3, 'o')
for i in range(1, 8):
	theta= i * pi / 4
	x1,y1,z1= TransP(TransT2(C),RotateR(theta),TransT1(C),P)
	x=x1/z1
	y=y1/z1
	plt.plot(x, y, 'o')
plt.savefig('Part2-1.jpg')
plt.show()
