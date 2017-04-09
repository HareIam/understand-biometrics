import numpy as np
from scipy import signal
A = np.matrix('1, 3, 2, 4;2, 2, 3, 4;5,5,4,5;8,9,0,1')
B = np.matrix('1, 2, 3, 4;2, 1, 3, 0;4,1,3,4;2,4,3,4')
C = signal.convolve(A, B)
print(C)
