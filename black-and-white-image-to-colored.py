#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 17:14:25 2018

@author: jordan
"""

import numpy as np
import cv2

testFgmask = np.array([[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20]])

def numberToTripleArray(number):
    return [number, number, number];

def blackAndWhiteImageToColored(image):
    rowNumber = image.shape[0]
    columnNumber = image.shape[1]
    flattenImage = image.flatten()
    
    vfunc = np.vectorize(flattenImage)
    return vfunc(flattenImage)
    
#    coloredImage = np.zeros((rowNumber, columnNumber, 3), dtype=np.uint8)


print(blackAndWhiteImageToColored(testFgmask))