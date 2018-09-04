#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 17:14:25 2018

@author: jordan
"""

import numpy as np
import cv2

def maskToMaskImage(mask):
    rowNumber = mask.shape[0]
    columnNumber = mask.shape[1]
    flattenImage = mask.flatten()
    coloredImage = np.array((flattenImage, flattenImage, flattenImage)).T

    return np.reshape(coloredImage, (rowNumber, columnNumber, 3))

def reformatColormap(colormap):
    colormapImage = np.zeros((256, 500, 3), dtype=np.uint8)
    for column in range(0, 500):
        colormapImage[:, column, :] = colormap[:, 0, :]

    return colormapImage

def displayColormap(colormap):
    colormapImage = reformatColormap(colormap)
    cv2.imwrite('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/colormap-image.png', colormapImage)
    image = cv2.imread('/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/results/colormap-image.png')
    cv2.imshow('image', image)
    if (cv2.waitKey(5) & 0xFF == ord('q')):
        cv2.destroyAllWindows()

def applyCustomColorMap(grayScaleImage, humanFlowColormap):
    reformatedGrayScaleImage = maskToMaskImage(grayScaleImage)
    coloredImage = cv2.LUT(reformatedGrayScaleImage, humanFlowColormap)

    return coloredImage;
