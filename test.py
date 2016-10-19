import matplotlib.pyplot as plt
import numpy as np
import mahotas as mh
from random import randint


def D(M1, M2):
    interm = np.sqrt(sum([sum([(M1[i][j]-M2[i, j])**2 for i in range(len(M1))]) for j in range(len(M1))]))
    return np.average(interm)  
    
image_elephant = mh.imread("/home/soso/Documents/cours/telecom/2_annee/ima201/elephant.jpg")
size_patch = 5
support = image_elephant.shape[:2]
L = 10
fz = np.array([[[0,0] for i in range(support[1]) ] for i in range(support[0])])

im_field = np.array([[ [0,0,0] for i in range(support[1])] for j in range(support[0])])
number_iteration = 2
patch = np.array([ [ [ [[0,0,0] for i in range(size_patch)] for i in range(size_patch)] for i in range(support[1])] for i in range(support[0])])                                                

#-------------phase initialisation----------------------------------------------------
for i in range(support[0]):
    for j in range(support[1]):
        x = randint(0, support[0]-1)
        y = randint(0, support[1]-1)
        fz[i][j][0] = x - i
        fz[i][j][1] = y - j
        im_field[i][j] = image_elephant[x][y]
        if ((i > size_patch/2 and i < support[0] - size_patch / 2 - 1) and (j > size_patch / 2 and j < support[1] - size_patch / 2 - 1)):
            patch[i][j] = image_elephant[i-size_patch//2:i+size_patch//2+1, j-size_patch//2:j+size_patch//2+1]
            
        
print("paaaaaatchMatch")
#-----------algorithme du patchMatch---------------------------------
fi = []
for turn in range(number_iteration):

    for i in range(size_patch//2, support[0]-size_patch//2-1):
        for j in range(size_patch//2, support[1]-size_patch//2-1):
            for k in range(L):
                x = 0
                y = 0
                while(x == y == 0):
                    x = randint(-1, 1)
                    y = randint(-1, 1)
                    f0 = fz[i][j]+(2**(k))*np.array([x, y])
                if(i+f0[0] in range(support[0]) and j+f0[1] in range(support[1])):
                    fi.append(f0)
            fi.append(fz[i][j])
            fz[i, j] = fi[np.argmin([D(patch[i, j], patch[i + f[0], j + f[1]]) for f in fi])]
            fi = []
    np.save("NNFelephant%s" % turn, fz)



