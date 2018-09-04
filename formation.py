#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

if __name__ == "__main__":
    repositoryPath = '/Users/jordan/Documents/Jordan/personal-projects/formation-theodo/'
    videoName = 'opencv-sample-video.avi'
    video = cv2.VideoCapture(repositoryPath + videoName)

    while(video.isOpened()):
        isFrame, frame = video.read()

        if (isFrame):
            cv2.imshow('video', frame)

            cv2.imwrite(repositoryPath + 'results/frame.png', frame)

            if (cv2.waitKey(5) & 0xFF == ord('q')):
                video.release()
                cv2.destroyAllWindows()
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
