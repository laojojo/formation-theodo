# -*- coding: utf-8 -*-

import cv2
from utils_colormap import applyCustomColorMap
from colormaps import humanFlowColormap

if __name__ == "__main__":
    repositoryPath = '/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/'
    videoName = 'opencv-sample-video.avi'
    video = cv2.VideoCapture(repositoryPath + videoName)
    backgroundSubstractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=128, detectShadows=False)
    alpha = 0.3
    maskTrace = 0
    imageCount = 0

    while(video.isOpened()):
        isFrame, frame = video.read()

        if (isFrame):
            mask = backgroundSubstractor.apply(frame, None, 0.01)

            maskTrace = maskTrace + mask

            heatMap = applyCustomColorMap(maskTrace, humanFlowColormap)
            frameCopy = frame.copy()

            result = cv2.addWeighted(heatMap, alpha, frameCopy, 1 - alpha, 0)

            cv2.imshow('result', result)

            print('Image number:', imageCount)

            cv2.imwrite(repositoryPath + 'results/frame.png', frame)
            cv2.imwrite(repositoryPath + 'results/mask.png', mask)
            cv2.imwrite(repositoryPath + 'results/maskTrace.png', maskTrace)
            cv2.imwrite(repositoryPath + 'results/heatMap.png', heatMap)
            cv2.imwrite(repositoryPath + 'results/result.png', result)

            imageCount += 1

            if (cv2.waitKey(5) == ord('q')):
                video.release()
                cv2.destroyAllWindows()
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
