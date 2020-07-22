import os
import cv2


def readVideo(filepath):
    assert os.path.exists(filepath),"file doesnt exist"
    video = cv2.VideoCapture(filepath)
    return video

def readFrame(video):
    status,frame = video.read()
    if (status):
        return frame
    else:
        print("could not find a frame ")
        return None

def displayFrame(frame):
    if frame is None:
        print("there was no frame")
        exit()
    cv2.imshow("frame",frame)
    #cv2.waitKey()

