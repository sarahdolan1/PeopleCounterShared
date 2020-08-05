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
import new_detections
from new_detections import centroid_tracker

def main():
    # start the frames per second throughput estimator
    fps = FPS().start()

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
    time.sleep(1.0)
    total_frames = 0

    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error Opening Video Stream Or File...")

    # loop over frames from the video stream
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        total_frames += 1
        ret, frame = cap.read()
        if ret == True:
            # resize the frame to have a maximum width of 500 pixels
            # (the less data we have, the faster we can process it),
            # then convert the frame from BGR to RGB for dlib
            frame = imutils.resize(frame, width=300)

            detections = get_detections(frame, net)
            centroid = centroid_tracker(detections, frame, net)
            new_detections.centroid_tracker(detections, frame, net)

            # movement = movementcounter(direction, centroid)

            # draw the label into the frame
            write_text(frame, " ", (20, 487), (0, 255, 0))

            # Display the resulting frame
            cv2.imshow("People Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            fps.update()

        # Press Q on keyboard to  exit
        if key == ord("q"):
            break

    # stop the timer and display FPS information

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # printdd("[INFO] approx. FPS: " )

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def get_detections(frame, net):

    # if the frame dimensions are empty, set them
    (height, width) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
    net.setInput(blob)
    detections = net.forward()

    return detections


def write_text(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)
    # initialize the video writer (we'll instantiate later if need be)
    writer = None
    # initialize the frame dimensions (set them as soon as we read the first frame from the vid)
    width = 300
    height = 200

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
    cv2.line(img, (width // 2, 0), (width // 2, height), (0, 0, 255), 2)

    # initialize the total number of frames processed thus far, along with
    # the total number of objects that have moved either Right or Left
    total_frames = 0
    total_left = 0
    total_right = 0

    status = 1

    # construct a tuple of information we will be displaying on the frame
    info = [
        ("Left", total_right),
        ("Right", total_left),
        ("Status", status),
    ]

    for (i, (k, value)) in enumerate(info):
        text = "{}: {}".format(k, value)
        cv2.putText(
            img,
            text,
            (10, 170 - ((i * 20) + 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            2,
        )


if __name__ == "__main__":
    main()
