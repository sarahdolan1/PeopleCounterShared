#these function will get detections and be a centroid tracker
'''def trackPeople():
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    return CLASSES
def centroidTracker():
    #find centroids of poeple to help track people
      # instantiate our centroid tracker, then initialize a list to store each of our dlib correlation
    # trackers, followed by a dictionary to map each unique object ID to a TrackableObject
     ct = CentroidTracker(maxDisappeared=40, maxDistance=400)
     trackers = []
     trackableObjects = {}
     STATUS = "Waiting"
     rects = []

     if TOTAL_FRAMES % args["skip_frames"] == 0:
        STATUS = "Detecting"
        trackers = []
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)

    else:
        for tracker in trackers:
            STATUS = "Tracking"
            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.top())
            startY = int(pos.left())
            endX = int(pos.bottom())
            endY = int(pos.right())
            rects.append((startX, startY, endX, endY))

    objects = ct.update(rects)
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

        trackableObjects[objectID] = to'''

#def getDetections():
#use blob to get the dections to detect the people
#return

#def centroidTracker():
#find centroids of poeple to help track people
#return

#def checkBoundary():
#used in future final code; irrelevant for now
#return