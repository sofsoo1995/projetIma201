

import matplotlib.pyplot as plt
import numpy as np

from random import randint
import cv2
import time

def D(M1, M2):
    # calcule de distance entre des patchs
    interm = np.linalg.norm(M1-M2)
    return interm

image = cv2.imread("TP_C02_007_copy.png", 0)
size_patch = (5, 5)
support = image.shape[:2]
L = 10
fz = np.zeros((support[0], support[1], 2))
number_iteration = 4
TD1 = 8

# -------------phase initialisation-------------------------
for i in range(support[0]):
    for j in range(support[1]):
        while(np.linalg.norm(fz[i, j], np.inf) < TD1):
            # But : eloigner suffisamment pour qu'il trouve des images similaires
            x = randint(0, support[0]-1)
            y = randint(0, support[1]-1)
            fz[i][j][0] = x - i
            fz[i][j][1] = y - j
        
print("paaaaaatchMatch")
# -----------algorithme du patchMatch---------------------------------
np.save("NNFrandom", fz)


def updateNNFRandom(fz, x, y):
    # permet de trouver le meilleur voisin par une propagation aleatoire
    M1 = cv2.getRectSubPix(image, size_patch, (i, j))
    distMin = D(M1,cv2.getRectSubPix(image, size_patch,(i+fz[i][j][0],j+fz[i][j][1])))
    fmin = fz[i, j]
    for k in range(L):
        x = 0
        y = 0
        while(x == y == 0):
            x = randint(-1, 1)
            y = randint(-1, 1)
        f0 = fz[i][j] + (2**(k))*np.array([x, y])
        if(i+f0[0] > 0 and i+f0[0] < support[0] and j + f0[1] > 0 and j+f0[1]<support[1]):
            M2 = cv2.getRectSubPix(image, size_patch, (i+f0[0], j+f0[1]))
            if(D(M1, M2) < distMin and np.linalg.norm(f0, np.inf) >= TD1):
                fmin = f0
                distMin = D(M1, M2)
    return fmin


def updateNNFProp(fz, x, y):
    # permet de construire le NNF iterativement en regardant les precedents
    M1 = cv2.getRectSubPix(image, size_patch, (i, j))
    distMin = D(M1,cv2.getRectSubPix(image, size_patch,(i+fz[i][j][0],j+fz[i][j][1])))
    fmin = fz[i, j]
    fi = []
    if (i > 0):
        fi.append(fz[i-1, j])
    if(j > 0):
        fi.append(fz[i, j-1])
    for v in fi:
        M2 = cv2.getRectSubPix(image, size_patch, (i + v[0], j + v[1]))
        if(D(M1, M2) < distMin):
            fmin = v
            distMin = D(M1, M2)
    return fmin


start = time.time()

for turn in range(number_iteration):

    for i in range(support[0]):
        for j in range(support[1]):
            fz[i, j] = updateNNFProp(fz, i, j)
            
            fz[i, j] = updateNNFRandom(fz, i, j)
    np.save("NNFim2%s" % turn, fz)

# post-processing--------------------------
end = time.time()
print(end-start)
magn = np.sqrt(fz[:, :, 0]**2 + fz[:, :, 1]**2)
plt.imshow(magn)
plt.colorbar()
plt.show()
