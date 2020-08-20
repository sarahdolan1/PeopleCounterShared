import argparse
import time
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
import dlib
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

def main():
    # passing arguments
    args = parse_arguments()
    # filepath to the video
    filepath = "videos/webcamfootage.mp4"
    trackers = []
    trackable_objects = {}
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    # load the video
    video = cv2.VideoCapture(filepath)
    # initialize the network only once
    net = load_network(args["prototxt"], args["model"])
    # iterating through the video
    total_frames = 0
    total_right = 0
    total_left = 0
    skip_frames = 10
    while video.isOpened():
        ret, frame = video.read()
        frame = imutils.resize(frame, width = 500)
        frame = write_text(frame, " leftyyyyy", (20, 230), (0, 255, 0))

        # if total_frames % (skip_frames) == 0:

        people_list = get_people(frame, net, trackers, min_confidence=0.4)

        #else:
        #    people_list = update_tracker(frame, trackers)
        frame = display_output(frame, people_list)
        cv2.imshow("image", frame)
        cv2.waitKey()


        right, left = count_people(ct, people_list, trackable_objects)
        total_right += right
        total_left += left
        total_frames += 1


        print("left", total_left)
        print("right", total_right)

def display_output(image, res, min_confidence=0.2):
    """
    Decode and display the output.
    Args:
            image: Input image.
            res: Network result.
    Returns:
            image: Image with boxes drawn on it.
    """
    for box in res:

        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]),
                        (232, 35, 244), 2)

    # Return image with boxes drawn on it.
    return image


"""
def display_output(image, res, min_confidence=0.2):

    Decode and display the output.
    Args:
            image: Input image.
            res: Network result.
    Returns:
            image: Image with boxes drawn on it.

    # Initialize boxes and classes.
    boxes, classes = {}, {}
    data = res[0][0]

    # Enumerate over all proposals.
    for number, proposal in enumerate(data):
        # For all proposals with confidence greater than 0.5.

        if proposal[2] > min_confidence:
            # Get index.
            imid = np.int(proposal[0])
            # Image height and width.
            ih, iw = image.shape[:-1]
            # Class label.
            label = np.int(proposal[1])
            # Output confidence.
            confidence = proposal[2]
            # Resize box predictions for input image.
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])
            # Add boxes and classes.
            if not imid in boxes.keys():
                boxes[imid] = []
                boxes[imid].append([xmin, ymin, xmax, ymax])
            if not imid in classes.keys():
                classes[imid] = []
                classes[imid].append(label)
    print(classes)
    print(boxes)
    # Draw boxes for all predictions.
    for imid in classes:
        for box in boxes[imid]:
            print("drawing box")
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]),
                          (232, 35, 244), 2)
    # Return image with boxes drawn on it.
    return image """


def load_network(prototxt_path, model_path):
    """we are trying to load a caffe network from disk
    Args:
            prototxt_path ([type]):filepath for the caffeProto
            model_path ([type]): Filepath for the caffemodel
    Returns:
            net: cv2.dnn network
    """

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net


def parse_arguments():
    """parses arguments
    Returns:
            args
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-p",
                    "--prototxt",
                    required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m",
                    "--model",
                    required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-i",
                    "--input",
                    type=str,
                    help="path to optional input video file")
    ap.add_argument("-o",
                    "--output",
                    type=str,
                    help="path to optional output video file")
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
    return args


def get_people(frame, net, trackers, min_confidence=0.4):
    """ detect people
    Args:
            frame (uint8): image
            net (cv2.dnn): neural network
            min_confidence (float): minimum for detection, defaults to 0.4
    Returns:
            people_list: list of bounding boxes containing people
    """

    classes = [
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
        "person",  # 15
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    people_list = []

    (height, width) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
    net.setInput(blob)
    detections = net.forward()
    ##we think its broken here ie doesnt go into the for loop
    status = "Detecting"

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by requiring a minimum
        # confidence
        if confidence > min_confidence:
            print(confidence)
            # extract the index of the class label from the
            # detections list
            idx = int(detections[0, 0, i, 1])
            print(idx, classes[idx])
            # if the class label is not a person, ignore it
            if classes[idx] == "person":
                print("Person Detected")
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                box = box.astype("int")
                people_list.append(box)

    print(people_list)
    for people in people_list:
        tracker = dlib.correlation_tracker()

        (start_x, start_y, end_x, end_y) = people
        rect = dlib.rectangle(start_x, start_y, end_x, end_y)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracker.start_track(rgb, rect)
        trackers.append(tracker)

    #box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
    #(start_x, start_y, end_x, end_y) = box.astype("int")

    # detections = entire list of detections; need to be threshholded; detections must be a person
    print(people_list)
    return people_list


def update_tracker(frame, trackers):
    """Track people over time
    Args:
            ct ([type]): dlib centroid tracker
            people_list ([type]): list of bounding boxes containing people
            frame ([type]): image
            net ([type]): neural network
    Returns:
            None
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    status = "Waiting"
    people_list = []

    for tracker in trackers:
        status = "Tracking"
        tracker.update(rgb)
        pos = tracker.get_position()

        # unpack the position object
        start_x = int(pos.top())
        start_y = int(pos.left())
        end_x = int(pos.bottom())
        end_y = int(pos.right())

        people_list.append((start_x, start_y, end_x, end_y))
    return people_list


def count_people(ct, people_list, trackable_objects):
    left = 0
    right = 0
    objects = ct.update(people_list)
    height = 500
    width = 500
    for (objectID, centroid) in objects.items():
        to = trackable_objects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)

        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:

                if direction < 0 and centroid[1] < width // 2:
                    right += 1
                    to.counted = True

                elif direction > 0 and centroid[1] > width // 2:
                    left += 1
                    to.counted = True

        trackable_objects[objectID] = to

    return right, left


def write_text(img, text, pos, bg_color):
    """[summary]
    Args:
            img ([type]): image
            text ([type]): overlay text
            pos ([type]): position of the text
            bg_color ([type]): text colour
    """
    height =500
    width = 500
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
    cv2.line(img, (width // 2, 0), (width // 2, height), (0, 0, 255), 2)

    """TEXT = "ID {}".format(objectID)
    cv2.putText(frame, TEXT, (centroid[1] - 10, centroid[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.circle(frame, (centroid[1], centroid[0]), 4, (255, 255, 255),
                -1)"""

    WRITER = None
    if WRITER is not None:
        WRITER.write(frame)

    return img



if __name__ == "__main__":
    main()
    
