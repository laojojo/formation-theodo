# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:13:53 2017

@author: jorlao
"""

import numpy as np
import os
import cv2

# fonction permettant de mettre en transparence tout ce qui est en blanc
# Entree : chemin complet de l'imagie a mettre en transparence
# Alpha est le coefficient de transparence. un objet est totalement opaque si la valeur alpha est au minimum (0)
# threshold avec THRESH_BINARY met a partir de 254 inclus tous les octets en blanc (255)

def grabcut(imagePath):
    (path, imageName) = os.path.split(imagePath)
    (shortname, extension) = os.path.splitext(imageName)
    src = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,254,255,cv2.THRESH_BINARY_INV)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/' + shortname + '_cut.png', dst)
    #return "C:/Users/jorlao/Documents/Machine_Learning/Resultats/" + shortname + "_cut.png"
    return '/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/' + shortname + '_cut.png'

def applyCustomColorMap(im_gray):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    lut[:, 0, 0] = [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,253,251,249,247,245,242,241,238,237,235,233,231,229,227,225,223,221,219,217,215,213,211,209,207,205,203,201,199,197,195,193,191,189,187,185,183,181,179,177,175,173,171,169,167,165,163,161,159,157,155,153,151,149,147,145,143,141,138,136,134,132,131,129,126,125,122,121,118,116,115,113,111,109,107,105,102,100,98,97,94,93,91,89,87,84,83,81,79,77,75,73,70,68,66,64,63,61,59,57,54,52,51,49,47,44,42,40,39,37,34,33,31,29,27,25,22,20,18,17,14,13,11,9,6,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    lut[:, 0, 1] = [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,254,252,250,248,246,244,242,240,238,236,234,232,230,228,226,224,222,220,218,216,214,212,210,208,206,204,202,200,198,196,194,192,190,188,186,184,182,180,178,176,174,171,169,167,165,163,161,159,157,155,153,151,149,147,145,143,141,139,137,135,133,131,129,127,125,123,121,119,117,115,113,111,109,107,105,103,101,99,97,95,93,91,89,87,85,83,82,80,78,76,74,72,70,68,66,64,62,60,58,56,54,52,50,48,46,44,42,40,38,36,34,32,30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0 ]
    lut[:, 0, 2] = [195,194,193,191,190,189,188,187,186,185,184,183,182,181,179,178,177,176,175,174,173,172,171,170,169,167,166,165,164,163,162,161,160,159,158,157,155,154,153,152,151,150,149,148,147,146,145,143,142,141,140,139,138,137,136,135,134,133,131,130,129,128,127,126,125,125,125,125,125,125,125,125,125,125,125,125,125,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126,126]
    im_color = cv2.LUT(im_gray, lut)

    return im_color;

def blackAndWhiteImageToColored(image):
    rowNumber = image.shape[0]
    columnNumber = image.shape[1]
    flattenImage = image.flatten()
    coloredImage = np.array((flattenImage, flattenImage, flattenImage)).T

    return np.reshape(coloredImage, (rowNumber, columnNumber, 3))

# lecture de la video
cap = cv2.VideoCapture('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/opencv-sample-video.avi')

# creation de l'objet qui permet de supprimer l'arriere plan
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=128, detectShadows=False)


# degre de transparence
alpha = 0.5

# variable qui va contenir la cumulation des masques
arr = 0


while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/frame.png', frame)
    # Calcul du masque
    fgmask = fgbg.apply(frame, None, 0.01)
    
    cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/fgmask.png', fgmask)
    
    arr = arr + fgmask
    
    arrayReformated = blackAndWhiteImageToColored(arr)
    heat = applyCustomColorMap(arrayReformated)
    cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/heat.png', heat)
    path = grabcut('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/heat.png')
    heat_cut = cv2.imread(path)
    # resultat
    output = frame.copy()
    
    cv2.addWeighted(heat_cut, alpha, output, 1 - alpha, 0, output)
    # Display the resulting frame
    cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/heatmap.png', output)
    cv2.imshow('heatmap', output)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)