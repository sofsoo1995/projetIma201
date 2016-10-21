import matplotlib.pyplot as plt
import numpy as np

from random import randint
import cv2

def D(M1, M2):
    ##calcule de distance entre des patchs
    interm = np.linalg.norm(M1-M2)
    return interm  
    
image = cv2.imread("TP_C01_001_copy.png", 0)
size_patch = (5, 5)
support = image.shape[:2]
L = 10
fz = np.zeros((support[0], support[1], 2))
number_iteration = 2
TD1 = 8

#-------------phase initialisation----------------------------------------------------
for i in range(support[0]):
    for j in range(support[1]):
        while(np.linalg.norm(fz[i, j], np.inf) < TD1):
            # But : eloigner suffisamment pour qu'il trouve des images similaires
            x = randint(0, support[0]-1)
            y = randint(0, support[1]-1)
            fz[i][j][0] = x - i
            fz[i][j][1] = y - j
        
print("paaaaaatchMatch")
#-----------algorithme du patchMatch---------------------------------
fi = []
for turn in range(number_iteration):

    for i in range(support[0]):
        for j in range(support[1]):
            for k in range(L):
                x = 0
                y = 0
                while(x == y == 0):
                    x = randint(-1, 1)
                    y = randint(-1, 1)
                    f0 = fz[i][j]+(2**(k))*np.array([x, y])
                if(i+f0[0] > 0 and i+f0[0]< support[0] and j+f0[1] >0 and j+f0[1]< support[1]):
                    fi.append(f0)
            fi.append(fz[i][j])
            M1 = cv2.getRectSubPix(image, size_patch, (i, j))
            fz[i,j] = np.argmin([D(M1,cv2.getRectSubPix(image, size_patch, (i+v[0], j+v[1]))) for v in fi])
            
            fi = []
    np.save("NNFim%s" % turn, fz)



