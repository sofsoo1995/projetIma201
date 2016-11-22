import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# nombre d'echantillons a garder
N = 30
fig = plt.figure()
fig2 = plt.figure()
a = fig.add_subplot(2, 2, 1)
b = fig.add_subplot(2, 2, 2)
c = fig.add_subplot(2, 2, 3)
d = fig.add_subplot(2, 2, 4)
e = fig2.add_subplot(2, 2, 3)
f = fig2.add_subplot(2, 2, 4)
g = fig2.add_subplot(2, 2, 1)
h0 = fig2.add_subplot(2, 2, 2)

a.set_title('histogramme des deplacements verticaux de f', fontsize=12)
b.set_title('histogramme des deplacements horizontaux de f', fontsize=12)

c.set_title('histogramme de la norme de f', fontsize=12)

d.set_title('histogramme en 2d de f', fontsize=12)

e.set_title('norme de f apres seuillage adapte', fontsize=12)
f.set_title('f apres ouverture  morphologique', fontsize=12)
g.set_title('image originale', fontsize=12)
h0.set_title('norme de f', fontsize=12)

# A modifier pour changer l'image.
# C'est pas propre c'est pas automatique mais bon.
name = 'guat30'
fz = np.load("./NNFguat.npy")
img = cv2.imread('ps_guat.png', 0)
fzi = fz[:, :, 0]
fzj = fz[:, :, 1]
magn = np.sqrt(fzi**2 + fzj**2)
print(magn.shape)


# But : obtenir l'histogramme en X et en Y
a.hist(fzi.ravel(), np.uint(np.max(fzi)), [np.min(fzi), np.max(fzi)])
b.hist(fzj.ravel(), np.uint(np.max(fzj)), [np.min(fzj), np.max(fzj)])
c.hist(magn.ravel(), np.uint(np.max(magn)-np.min(magn)), [np.min(magn),
                                                          np.max(magn)])
d.hist2d(fzi.ravel(), fzj.ravel(), bins=40)
histfz, xedges, yedges, Image = plt.figure().add_subplot(1, 1, 1).hist2d(
    fzi.ravel(), fzj.ravel(),
    bins=[np.max(fzi)-np.min(fzi),
          np.max(fzj)-np.min(fzj)])


shape = histfz.shape


# Mettre les petits deplacement a 0.
for i in range(-8, 9):
    for j in range(-8, 9):
        histfz[shape[0]//2 + i, shape[1]//2 + j] = 0

# On selectionne les pics de l'histogramme 2d
size = len(histfz.reshape(-1))
percent = 100*float((size - N)/float(size))
print(percent)
mainDispla = np.where(histfz >= np.percentile(histfz, percent))
print(len(mainDispla[0]))
newFz = np.zeros(magn.shape)
for i in range(len(mainDispla[0])):
    newFz = np.logical_or(newFz, np.logical_and(
        fzi == xedges[mainDispla[0][i]],
        fzj == yedges[mainDispla[1][i]]))

newFz = 255 * np.uint8(newFz)

e.imshow(newFz)

# Morphologie mathematique pour enlever le bruit
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
filteredNewFz = cv2.morphologyEx(newFz, cv2.MORPH_OPEN, kernel)
f.imshow(filteredNewFz)

# On affiche les images
g.imshow(img, cmap=plt.get_cmap('gray'))
h0.imshow(magn)
plt.imsave('mask'+name+'.png', filteredNewFz)
fig.suptitle('differents histogramme de f pour le '+name)
fig2.suptitle('differentes etape de traitement pour le'+name)
fig.savefig('histo'+name+'.png')
fig2.savefig('etape_'+name+'.png')
