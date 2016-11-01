import numpy as np
import cv2
import matplotlib.pyplot as plt

fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
fz = np.load("./NNFphare.npy")
magn = np.sqrt(fz[:, :, 0]**2 + fz[:, :, 1]**2)
magnfilt = cv2.medianBlur(magn, 3)
plt.imshow(magn)
a = fig.add_subplot(1, 2, 2)
plt.imshow(magnfilt)
plt.show()
