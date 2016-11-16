import numpy as np

import matplotlib.pyplot as plt
import time


fig = plt.figure()
a = fig.add_subplot(2, 2, 1)
b = fig.add_subplot(2, 2, 2)
c = fig.add_subplot(2, 2, 3)
d = fig.add_subplot(2, 2, 4)
fz = np.load("./NNFphare.npy")
fzi = fz[:, :, 0]
fzj = fz[:, :, 1]
magn = np.sqrt(fzi**2 + fzj**2)
print(np.min(fzi))
print(np.max(fzi))
# But : obtenir l'histogramme en X et en Y
a.hist(fzi.ravel(), np.uint(np.max(fzi)), [np.min(fzi), np.max(fzi)])
b.hist(fzj.ravel(), np.uint(np.max(fzj)), [np.min(fzj), np.max(fzj)])
c.hist(magn.ravel(), np.uint(np.max(magn)-np.min(magn)), [np.min(magn),
                                                          np.max(magn)])
histp = d.hist2d(fzi.ravel(), fzj.ravel(), [np.uint(np.max(fzi)),
                                            np.uint(np.max(fzj))],
                 [[np.min(fzi), np.max(fzi)], [np.min(fzj), np.max(fzj)]])

plt.show()
