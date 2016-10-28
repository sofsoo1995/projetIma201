import matplotlib.pyplot as plt
import numpy as np

from random import randint
import cv2
import time

from patchmatch import *
# -------------phase initialisation-------------------------

image = cv2.imread("TP_C02_007_copy.png", 0)
size_patch = (15, 15)
support = image.shape[:2]
L = 10

number_iteration = 6
TD1 = 8

# -------------execution du code--------------------
fz = init_nnf(support, TD1)
print("patchMatch")
start = time.time()
k = np.zeros((4, 4))

fz = calcul_nnf(number_iteration, support, fz, image, size_patch, L, TD1)

end = time.time()

# affichage du resultat
print(end-start)
magn = np.sqrt(fz[:, :, 0]**2 + fz[:, :, 1]**2)
plt.imshow(magn)
plt.colorbar()
plt.show()

