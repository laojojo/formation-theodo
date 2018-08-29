# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:13:53 2017

@author: jorlao
"""

import cv2
from utils_colormap import maskToMaskImage, applyCustomColorMap
from colormaps import humanFlowColormap


# fonction permettant de mettre en transparence tout ce qui est en blanc
# Entree : chemin complet de l'imagie a mettre en transparence
# Alpha est le coefficient de transparence. un objet est totalement opaque si la valeur alpha est au minimum (0)
# threshold avec THRESH_BINARY met a partir de 254 inclus tous les octets en blanc (255)


if __name__ == "__main__":
    # lecture de la video
    cap = cv2.VideoCapture('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/opencv-sample-video.avi')
    
    # creation de l'objet qui permet de supprimer l'arriere plan
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=128, detectShadows=False)
    
    # degre de transparence
    alpha = 0.3
    
    # variable qui va contenir la cumulation des masques
    maskTrace = 0
    
    isFirstIteration = True
    
    imageCount = 0 ## Max 793

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/frame.png', frame)
            # Calcul du masque
    
            fgmask = fgbg.apply(frame, None, 0.01)
            cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/fgmask.png', fgmask)
    
            if isFirstIteration:
                isFirstIteration = False
                continue
    
            maskTrace = maskTrace + fgmask
            cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/maskTrace.png', maskTrace)
    
            arrayReformated = maskToMaskImage(maskTrace)
            heat = applyCustomColorMap(arrayReformated, humanFlowColormap)
            cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/heat.png', heat)
    
            # resultat
            output = frame.copy()
    
            cv2.addWeighted(heat, alpha, output, 1 - alpha, 0, output)
            # Display the resulting frame
            cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/heatmap.png', output)
            cv2.imshow('heatmap', output)
            print(imageCount)
            imageCount+=1
            if ((cv2.waitKey(5) & 0xFF == ord('q'))):
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                break
        else:
            break
