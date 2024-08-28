# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:38:39 2024

@author: lcuev
"""
from scipy.signal import convolve
import matplotlib.image as mpimg
import numpy as np
from tqdm import trange

vert = [[1,0,-1],[1,0,-1],[1,0,-1]]
hori = [[1,1,1],[0,0,0],[-1,-1,-1]]
dial = [[0,-1,-1],[1,0,-1],[1,1,0]]
diar = [[1,1,0],[1,0,-1],[0,-1,-1]]
cdex = ['|','—','\\','/'] 

def compress(img, bin_size=2):
    img = np.array(img)
    X,Y = img.shape[0] // bin_size, 2*img.shape[1] // bin_size 
    img = img[:X*bin_size,:Y*bin_size//2]
    shape = (X,bin_size,Y,bin_size//2)       
    
    return img.reshape(shape).mean(-1).mean(1)


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
            if m > 0.5:
                chars[x][y] = cdex[stack.index(m)]
                
    return chars

img_pref = 'frames/'
out_pref = 'chars/'
for n in trange(1,151):
    img_name = f'frame{n:04d}' + '.png'
    img_path = img_pref + img_name
    out_path = out_pref + img_name
    
    img = compress(mpimg.imread(img_path, 0))
    chars = to_chars(img)
    with open(out_path,'w') as out_file:
        for line in chars:
            out_file.write(''.join(line) + '\n')




