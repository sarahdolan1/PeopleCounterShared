import numpy as np
import cv2
import dlib
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

def update_tracker(ct, detections, frame, net):
    # find centroid of poeple to help track people
    # instantiate our centroid tracker, then initialize a list to store each of our dlib correlation
    # trackers, followed by a dictionary to map each unique object ID to a TrackableObject

    # resize the frame to have a maximum width of 500 pixels
    # (the less data we have, the faster we can process it),
    # then convert the frame from BGR to RGB for dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    # initialize the current status along with our list of
    # bounding box rectangles returned by either
    # (1) our object detector or (2) the correlation trackers
    status = "Waiting"
    rects = []
    trackable_objects = {}

    total_frames = 0
    total_left = 0
    total_right = 0
    skip_frames = 40
    # check to see if we should run a more computationally expensive object
    #  detection method to aid our tracker
    if total_frames % (skip_frames) == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []


                # construct a dlib rectangle object from the bounding box
                # coordinates and then start the dlib correlation tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                print("Person Detected")
                tracker.start_track(rgb, rect)
                # add the tracker to our list of trackers so we can utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the STATUS of our system to be 'tracking' rather than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()
            print(pos)

            # unpack the position object
            start_x = int(pos.top())
            start_y = int(pos.left())
            end_x = int(pos.bottom())
            end_y = int(pos.right())

            # add the bounding box coordinates to the rectangles list
            rects.append((start_x, start_y, end_x, end_y))

    # use the centroid tracker to associate the
    # (1) old object centroid with
    # (2) the newly computed object centroid
    objects = ct.update(rects)

    # loop over the tracked objects
    for (object_identification, centroid) in objects.items():
        print ("Right")
        # check to see if a trackable object exists for the current object ID
        to = trackable_objects.get(object_identification, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(object_identification, centroid)


        # otherwise, there is a trackable object so we can utilize it to determine direction
        else:

            # the difference between the x-coordinate of the *current* centroid
            # and the mean of *previous* centroid will tell
            # us in which direction the object is moving
            # (negative for 'Right' and positive for 'Left')

            y_coordinate = [c[1] for c in to.centroid]
            direction = centroid[1] - np.mean(y_coordinate)
            to.centroid.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving Right) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < width // 2:
                    total_right += 1
                    to.counted = True




                # if the direction is positive (indicating the object
                # is moving Left) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > width // 2:
                    total_left += 1
                    to.counted = True

            # store the trackable object in our dictionary
            trackable_objects[object_identification] = to
