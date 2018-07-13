# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:13:53 2017

@author: jorlao
"""

import numpy as np
import sys
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


# lecture de la video
cap = cv2.VideoCapture('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/opencv-sample-video.avi')

# creation de l'objet qui permet de supprimer l'arriere plan
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=128, detectShadows=False)


# degre de transparence
alpha = 0.5

# variable qui va contenir la cumulation des masques
arr = 0


while(1 & cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/frame.png', frame)
    # Calcul du masque
    fgmask = fgbg.apply(frame, None, 0.01)
    cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/fgmask.png', fgmask)
    
    arr = arr + fgmask
    
    heat = cv2.applyColorMap(arr, cv2.COLORMAP_HOT)
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