import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('/Users/jimmy/Desktop/RGBTrainChess/test_result/chessResult_000001.txt', skiprows = 4)

pred = data[:,[2,3,4]];

gd = data[:,[5,6,7]];

dif = pred - gd;

plt.figure()
plt.hist(dif[:,0],200)
plt.title('X')

plt.figure()
plt.hist(dif[:,1],200)
plt.title('Y')

plt.figure()
plt.hist(dif[:,2],200)
plt.title('Z')

plt.show()

