

import matplotlib.pyplot as plt
import numpy as np

from random import randint
import cv2
import time


def D(M1, M2):
    # calcule de distance entre des patchs
    interm = np.linalg.norm(M1-M2)
    return interm


def init_nnf(support, TD1):
    fz = np.zeros((support[0], support[1], 2))
    
    Mrand = np.zeros((support[0], support[1], 2))
    Mrand1 = np.random.randint(0, support[0], (support[0], support[1]))
    Mrand2 = np.random.randint(0, support[1], (support[0], support[1]))
    Mrand[:, :, 0] = Mrand1
    Mrand[:, :, 1] = Mrand2
    I = [[np.array([i, j]) for j in range(support[1])] for i in range(support[0])]
    I = np.array(I)
    fz = Mrand - I
    list_i, list_j = np.where((np.abs(fz[:, :, 0]) < TD1) | (np.abs(fz[:, :, 1]) < TD1))
    while(len(list_i) > 0):
        for i in range(len(list_i)):
            x = np.random.randint(0, support[0])
            y = np.random.randint(0, support[1])
            fz[list_i[i], list_j[i]][0] = x - list_i[i]
            fz[list_i[i], list_j[i]][1] = y - list_j[i]
        list_i, list_j = np.where((np.abs(fz[:, :, 0]) < TD1) | (np.abs(fz[:, :, 1]) < TD1))
    return fz
        

# -----------algorithme du patchMatch---------------------------------


def updateNNFRandom(fz, s1, s2, image, size_patch, L, TD1, support):
    # permet de trouver le meilleur voisin par une propagation aleatoire
    
    beg = time.time()
    M1 = cv2.getRectSubPix(image, size_patch, (s2, s1))
    phi = (s2+fz[s1][s2][1], s1+fz[s1][s2][0])
    M = cv2.getRectSubPix(image, size_patch, phi)
    
    distMin = D(M1, M)
   
    fmin = fz[s1, s2]
    
    for k in range(L):
        r = np.random.randint(-1, 2, 2)
 
        while(np.linalg.norm(r) == 0):
            r = np.random.randint(-1, 2, 2)
        f0 = fz[s1][s2] + (2**(k))*r
        if(s1+f0[0] > 0 and s1+f0[0] < support[0] and s2 + f0[1] > 0 and s2+f0[1]<support[1]):
            M2 = cv2.getRectSubPix(image, size_patch, (s2+f0[1], s1+f0[0]))
            dist = D(M1, M2)
            if(dist < distMin and np.linalg.norm(f0, np.inf) >= TD1):
                fmin = f0
                distMin = dist
    end = time.time()
    
    return fmin


def updateNNFProp(fz, s1, s2, image, size_patch):
    # permet de construire le NNF iterativement en regardant les precedents
    M1 = cv2.getRectSubPix(image, size_patch, (s2, s1))
    distMin = D(M1, cv2.getRectSubPix(image, size_patch, (s2+fz[s1][s2][1], s1+fz[s1][s2][0])))
    fmin = fz[s1, s2]
    fi = []
    if (s1 > 0):
        fi.append(fz[s1-1, s2])
    if(s2 > 0):
        fi.append(fz[s1, s2-1])
    for v in fi:
        M2 = cv2.getRectSubPix(image, size_patch, (s2 + v[1], s1 + v[0]))
        if(D(M1, M2) < distMin):
            fmin = v
            distMin = D(M1, M2)
    return fmin


def calcul_nnf(number_iteration, support, fz, image, size_patch, L, TD1):
    for turn in xrange(number_iteration):

        for j in xrange(0, support[1]-1):
            for i in xrange(0, support[0]-1):
                fz[i, j] = updateNNFProp(fz, i, j, image, size_patch)
                fz[i, j] = updateNNFRandom(fz, i, j, image, size_patch, L, TD1, support)
    return fz



