# -*- coding: utf-8 -*-

import cv2
from utils_colormap import applyCustomColorMap
from colormaps import humanFlowColormap

if __name__ == "__main__":
    repositoryPath = '/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/'
    videoName = 'opencv-sample-video.avi'
    video = cv2.VideoCapture(repositoryPath + videoName)
    backgroundSubstractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=128, detectShadows=False)
    maskTrace = 0

    while(video.isOpened()):
        isFrame, frame = video.read()

        if (isFrame == True):
            mask = backgroundSubstractor.apply(frame, None, 0.01)

            maskTrace = maskTrace + mask

            heat = applyCustomColorMap(maskTrace, humanFlowColormap)

            cv2.imshow('heat', heat)

            cv2.imwrite(repositoryPath + 'results/frame.png', frame)
            cv2.imwrite(repositoryPath + 'results/mask.png', mask)
            cv2.imwrite(repositoryPath + 'results/maskTrace.png', maskTrace)
            cv2.imwrite(repositoryPath + 'results/heat.png', heat)

            if (cv2.waitKey(5) & 0xFF == ord('q')):
                video.release()
                cv2.destroyAllWindows()
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
