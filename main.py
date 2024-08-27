# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:38:39 2024

@author: lcuev
"""
from scipy.signal import convolve
import matplotlib.image as mpimg
import numpy as np

vert = [[1,0,-1],[1,0,-1],[1,0,-1]]
hori = [[1,1,1],[0,0,0],[-1,-1,-1]]
dial = [[0,-1,-1],[1,0,-1],[1,1,0]]
diar = [[1,1,0],[1,0,-1],[0,-1,-1]]

def stretch(img,factor=2):
    return [[img[i][j//2] for j in range(len(img[0])*2)] for i in range(len(img))]

def compress(img, bin_size=1):
    img = np.array(img)
    X,Y = img.shape[0] // bin_size, img.shape[1] // bin_size
    img = img[:X*bin_size,:Y*bin_size]
    shape = (X,bin_size,Y,bin_size)        
    return img.reshape(shape).mean(-1).mean(1)

img_path = 'felipe.png'
img = compress(stretch(mpimg.imread(img_path, 0).mean(axis=2)))
cdex = ['|','—','\\','/'] 

def to_chars(img):
    
    ivert = convolve(img,vert)
    ihori = convolve(img,hori)
    idial = convolve(img,dial)
    idiar = convolve(img,diar)
    X,Y = ivert.shape
    chars = [[' ' for j in range(Y)] for i in range(X)]
    for x in range(X):
        for y in range(Y):
            stack = [abs(ivert[x][y]),abs(ihori[x][y]),abs(idial[x][y]),abs(idiar[x][y])]
            m = max(stack)
            if m > 0.2:
                chars[x][y] = cdex[stack.index(m)]
    return chars


chars = to_chars(img)
for col in chars:
    print(''.join(col))

