import numpy as np
import cv2
from imutils.video import FPS
import dlib
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

def get_neural_network(proto_path, model_path):
    """" load a caffe network from disk
    Args:
            prototxt_path ([type]):filepath for the caffeProto
            model_path ([type]): Filepath for the caffemodel
    Returns:
            net: cv2.dnn network
    """
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    return net

def get_detections(net, frame, width, height):
    """ detect people
    Args:
            net (cv2.dnn): neural network
            frame (uint8): image
            width (length): width of video
            height (length): height of video
    Returns:
            detections, status: detecting any class, stage of tracking
    """
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
    net.setInput(blob)
    detections = net.forward()
    status = "Detecting"
    return detections, status

def get_people(detections, classes, width, height, given_confidence):
    """ detect people
    Args:
            detections ([type]): detecting any class
            classes ([type]): classes of identification
            width (length): width of video
            height (length): height of video
            given_confidence (float): minimum for detection, defaults to 0.4
    Returns:
            people_list: list of bounding boxes containing people
    """
    people_list = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > given_confidence:
            idx = int(detections[0, 0, i, 1])
            if classes[idx] != "person":
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            rect = dlib.rectangle(start_x, start_y, end_x, end_y)
            people_list.append(rect)
    return people_list

def track_people(trackers, rgb):
    """ how to track a person
    Args:
            trackers ([type]): track movements
            rgb ([type]):  red, green & blue
    Returns:
            rects, status: rects, stage of tracking
    """
    rects = []
    for tracker in trackers:
        tracker.update(rgb)
        pos = tracker.get_position()
        start_x = int(pos.top())
        start_y = int(pos.left())
        end_x = int(pos.bottom())
        end_y = int(pos.right())
        rects.append((start_x, start_y, end_x, end_y))
    status = "Tracking"
    return rects, status

def people_counter(video_capture, net, ct, classes,
                   output_video, fourcc, total_frames, total_left,
                   total_right, trackable_objects, confidence, skip_frames):
    """ Track people over time
    Args:
           video_capture ([type]): video
           net (cv2.dnn): neural network
           ct ([type]): dlib centroid tracker
           classes ([type]): classes of identification
           output_video ([type]): what the output video is saved under
           fourcc ([type]): video writer
           total_frames (float): complete number of frames
           total_left (float): complete number of lefts
           total_right (float): complete number of rights
           trackable_objects ([type]): list of objects on screen from classes
           confidence (float): minimum for detection, defaults to 0.4
           skip_frames (float): defaults to 30
    Returns:
           fps.fps(): frames per second
    """
    fps = FPS().start()
    while True:
        ok, frame = video_capture.read()
        if frame is None: # END of Video Loop
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        (height, width) = frame.shape[:2]
        writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height), True)
        status = "Waiting"
        rects = []

        videotime = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if total_frames % skip_frames == 0:
            trackers = []

            detections, status = get_detections(net, frame, width, height)
            people_list = get_people(detections, classes, width, height, confidence)

            for rect in people_list:
                tracker = dlib.correlation_tracker()
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
        else:
            rects, status = track_people(trackers, rgb)

        write_on_frame_line(frame, width, height)
        objects = ct.update(rects)

        for (object_identification, centroid) in objects.items():
            to = trackable_objects.get(object_identification, None)
            if to is None:
                to = TrackableObject(object_identification, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[0] - np.mean(y)
                to.centroids.append(centroid)
                if not to.counted:
                    if direction < 0 and centroid[1] < width // 2:
                        total_left += 1
                        to.counted = True
                    elif direction > 0 and centroid[1] > width // 2:
                        total_right += 1
                        to.counted = True
            trackable_objects[object_identification] = to
            write_on_frame_object(object_identification, centroid, frame)

        write_on_frame_legend(total_left, total_right, videotime, frame, height)
        writer.write(frame)

        cv2.imshow("People Counter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        total_frames += 1
        fps.update()

    fps.stop()
    cv2.destroyAllWindows()

    return fps.fps()

def write_on_frame_line(frame, width, height):
    """
    Display the output
    Args:
        frame(uint8) : Input Image
        width(length) : width of frame
        height(length): height of frame

    Returns:
        none

    """
    cv2.line(frame, (width // 2, 0), (width // 2, height),
             (0, 0, 255), 2) # To change the dividing Line
    return None

def write_on_frame_object(object_identification, centroid, frame):
    """
    Display the output
    Args:
         frame(uint8) : Input Image
         object_identification([type]) :the number given to identify the object
         centriod ([type]):the centerpoint of the object

    Returns:
        none

    """
    text = "ID {}".format(object_identification)
    cv2.putText(frame, text, (centroid[1] - 10, centroid[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.circle(frame, (centroid[1], centroid[0]), 4, (255, 255, 255), -1)
    return None

def write_on_frame_legend(total_left, total_right,
                          videotime, frame, height):
    """
    Display the output
    Args:
        total_left(float): total number of lefts
        total_right(float): total number of right
        videotime([type]):how long the video the video has played for
        frame(uint8):Input Image
        height(length): height of the frame

    Returns:
        none

    """
    info = [("Left", total_left), ("Right", total_right),
            ("Time", "{:.2f}".format(videotime)),] #("Status", status) - to display status
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return None

def main(input_video, output_video, proto_path, model_path):

    classes = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    video_capture = cv2.VideoCapture(input_video)
    net = get_neural_network(proto_path, model_path)
    ct = CentroidTracker(maxDisappeared=40, maxDistance=500)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    trackable_objects = {}
    total_frames = 0
    total_right = 0
    total_left = 0
    confidence = 0.4
    skip_frames = 30

    fps = people_counter(video_capture, net, ct, classes,
                         output_video, fourcc, total_frames, total_left,
                         total_right, trackable_objects, confidence, skip_frames)

    print("Approx. FPS: {:.2f}".format(fps))
    return None

input_video = 'videos/final_vid.mp4'
output_video = 'output/output_final_vid.avi'
proto_path = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
model_path = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'

main(input_video, output_video, proto_path, model_path)
 
