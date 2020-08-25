from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def getNN(protoPath, modelPath):
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    return net


def getDetections(net, frame, W, H):
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()
    status = "Detecting"
    return detections, status


def getPeople(detections, CLASSES, W, H, givenConfidence):
    peopleList = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > givenConfidence:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            rect = dlib.rectangle(startX, startY, endX, endY)
            peopleList.append(rect)
    return peopleList


def trackPeople(trackers, rgb):
    rects = []
    for tracker in trackers:
        tracker.update(rgb)
        pos = tracker.get_position()
        startX = int(pos.top())
        startY = int(pos.left())
        endX = int(pos.bottom())
        endY = int(pos.right())
        rects.append((startX, startY, endX, endY))
    status = "Tracking"
    return rects, status


def peopleCounter(videoCapture, net, ct, CLASSES, output_video, fourcc, totalFrames, totalLeft, totalRight, trackableObjects, confidence, skipFrames):
    fps = FPS().start()
    while True:
        ok, frame = videoCapture.read()
        if frame is None: # END of Video Loop
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        (H, W) = frame.shape[:2]
        writer = cv2.VideoWriter(output_video, fourcc, 30, (W, H), True)
        status = "Waiting"
        rects = []

        videotime = videoCapture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if totalFrames % skipFrames == 0:
            trackers = []

            detections, status = getDetections(net, frame, W, H)
            peopleList = getPeople(detections,CLASSES, W, H, confidence)

            for rect in peopleList:
                tracker = dlib.correlation_tracker()
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
        else:
            rects, status = trackPeople(trackers, rgb)

        writeonFrame_Line(frame, W, H)
        objects = ct.update(rects)

#         UpdateCount(objects, frame, H, totalLeft, totalRight, trackableObjects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                if not to.counted:
                    if direction < 0 and centroid[1] < W // 2:
                        totalLeft += 1
                        to.counted = True
                    elif direction > 0 and centroid[1] > W // 2:
                        totalRight += 1
                        to.counted = True
            trackableObjects[objectID] = to
            writeonFrame_Object(objectID, centroid, frame)

        writeonFrame_Legend(totalLeft, totalRight, videotime, frame, W, H, status)
        writer.write(frame)

        cv2.imshow("People Counter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        totalFrames += 1
        fps.update()

    fps.stop()
    cv2.destroyAllWindows()

    return fps.fps()


def writeonFrame_Line(frame, W, H):
    cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 0, 255), 2) # To change the dividing Line
    return None

def writeonFrame_Object(objectID, centroid, frame):
    text = "ID {}".format(objectID)
    cv2.putText(frame, text, (centroid[1] - 10, centroid[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.circle(frame, (centroid[1], centroid[0]), 4, (255, 255, 255), -1)
    return None

def writeonFrame_Legend(totalLeft, totalRight, videotime, frame, W, H, status):
    info = [("Left", totalLeft),("Right", totalRight),("Time", "{:.4f}".format(videotime)),] #("Status", status) - to display status
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return None


def main(input_video, output_video, protoPath, modelPath):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
    videoCapture = cv2.VideoCapture(input_video)
    net = getNN(protoPath, modelPath)
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    trackableObjects = {}
    totalFrames = 0
    totalRight = 0
    totalLeft = 0
    confidence = 0.4
    skipFrames = 30

    fps = peopleCounter(videoCapture, net, ct, CLASSES, output_video, fourcc, totalFrames, totalLeft,
            totalRight, trackableObjects, confidence, skipFrames)

    print("Approx. FPS: {:.2f}".format(fps))
    return None


input_video = './videos/webcamfootage.mp4'
output_video = './output/output_webcamfootage.avi'
protoPath = './mobilenet_ssd/MobileNetSSD_deploy.prototxt'
modelPath = './mobilenet_ssd/MobileNetSSD_deploy.caffemodel'

main(input_video, output_video, protoPath, modelPath)



# def updateCount(objects, frame, H, totalLeft, totalRight, trackableObjects):
#     for (objectID, centroid) in objects.items():
#         to = trackableObjects.get(objectID, None)
#         if to is None:
#             to = TrackableObject(objectID, centroid)
#         else:
#             y = [c[1] for c in to.centroids]
#             direction = centroid[1] - np.mean(y)
#             to.centroids.append(centroid)
#             if not to.counted:
#                 if direction < 0 and centroid[1] < H // 2:
#                     totalLeft += 1
#                     to.counted = True
#                 elif direction > 0 and centroid[1] > H // 2:
#                     totalRight += 1
#                     to.counted = True
#         trackableObjects[objectID] = to
#         writeonFrame_Object(objectID, centroid, frame)
#     return None
