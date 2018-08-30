# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:13:53 2017

@author: jorlao
"""

import cv2
from utils_colormap import maskToMaskImage, applyCustomColorMap
from colormaps import humanFlowColormap


if __name__ == "__main__":
    repositoryPath = '/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/'
    video = cv2.VideoCapture(repositoryPath + 'opencv-sample-video.avi')
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=128, detectShadows=False)
    alpha = 0.3
    maskTrace = 0
    imageCount = 0

    while(video.isOpened()):
        ret, frame = video.read()

        if ret == True:
            mask = fgbg.apply(frame, None, 0.01)

            maskTrace = maskTrace + mask

            reformatedMaskTrace = maskToMaskImage(maskTrace)
            heat = applyCustomColorMap(reformatedMaskTrace, humanFlowColormap)
            heatmap = frame.copy()

            cv2.addWeighted(heat, alpha, heatmap, 1 - alpha, 0, heatmap)

            cv2.imshow('heatmap', heatmap)

            print('Image number:', imageCount)

            cv2.imwrite(repositoryPath + 'results/frame.png', frame)
            cv2.imwrite(repositoryPath + 'results/mask.png', mask)
            cv2.imwrite(repositoryPath + 'results/maskTrace.png', maskTrace)
            cv2.imwrite(repositoryPath + 'results/heat.png', heat)
            cv2.imwrite(repositoryPath + 'results/heatmap.png', heatmap)

            imageCount += 1

            if (cv2.waitKey(5) & 0xFF == ord('q')):
                video.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
