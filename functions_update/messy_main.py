# import the necessary packages
import argparse
import time
import numpy as np
import imutils
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import dlib
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

from detection import centroidTracker

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file"
    )
    ap.add_argument(
        "-m", "--model", required=True, help="path to Caffe pre-trained model"
    )
    ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
    ap.add_argument(
        "-o", "--output", type=str, help="path to optional output video file"
    )
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.4,
        help="minimum probability to filter weak detections",
    )
    ap.add_argument(
        "-s",
        "--skip-frames",
        type=int,
        default=30,
        help="# of skip frames between detections",
    )
    args = vars(ap.parse_args())

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    webcamfootage = "videos/webcamfootage.mp4"
    cap = cv2.VideoCapture(webcamfootage)
    print("Opening Video Stream Or File...")

    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error Opening Video Stream Or File...")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            detections = Get_Detections(frame, net)

            # draw the label into the frame
            write_text(frame, "TOTAL LEFT", (20, 600), (0, 255, 0))
            write_text(frame, "TOTAL RIGHT ", (20, 625), (0, 0, 225))
            write_text(frame, "STATUS ", (20, 650), (255, 0, 0))

            # Display the resulting frame
            cv2.imshow("People Counter", frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def Centroid_Tracker(detections):
  # each of our dlib correlation trackers, followed by a dictionary to
  # map each unique object ID to a TrackableObject
  ct = CentroidTracker(maxDisappeared=40, maxDistance=400)
  trackers = []
  trackableObjects = {}


  # construct a dlib rectangle object from the bounding box coordinates and then start the dlib correlation tracker
  tracker = dlib.correlation_tracker()
  rect = dlib.rectangle(startX, startY, endX, endY)
  tracker.start_track(rgb, rect)

  # add the tracker to our list of trackers so we can utilize it during skip frames
  trackers.append(tracker)

  tracker.update(rgb)
  pos = tracker.get_position()

  # unpack the position object
  startX = int(pos.top())
  startY = int(pos.left())
  endX = int(pos.bottom())
  endY = int(pos.right())

  # add the bounding box coordinates to the rectangles list
  rects.append((startX, startY, endX, endY))
  return centroids


def Get_Detections(frame, net):
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()

    return detections


def write_text(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)
    WRITER = None
    W = 1300
    H = 1000

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
    cv2.line(img, (W // 2, 0), (W // 2, H), (0, 0, 255), 2)


if __name__ == "__main__":
    main()
