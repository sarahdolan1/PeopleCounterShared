import os
import cv2
import imutils
import numpy as np

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

def onscreendraw(frame):
  W = None
  H = None
  frame = imutils.resize(frame, width=500)
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#-------------------------------------------------------------------------

def words(frame):
  frame = cv2.imread('webcamfootage.mp4',cv2.IMREAD_UNCHANGED)

  position = ((int) (200), (int) (200))

  cv2.putText(
      frame, #numpy array on which text is written
      "Python Examples", #text
      position, #position at which writing has to start
      cv2.FONT_HERSHEY_SIMPLEX, #font family
      1, #font size
      (209, 80, 0, 255), #font color
      3) #font stroke
  cv2.imwrite('output.mp4', frame)
  #VIDEO
#output

'''
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 0, 255), 2)
            if not to.counted:
                if direction < 0 and centroid[1] < W // 2:
                    TOTAL_RIGHT += 1
                    to.counted = True


                elif direction > 0 and centroid[1] > W // 2:
                    TOTAL_LEFT += 1
                    to.counted = True

        TEXT = "ID {}".format(objectID)
        cv2.putText(frame, TEXT, (centroid[1] - 10, centroid[0] -10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (centroid[1], centroid[0]), 4, (255, 255, 255), -1)

    info = [
        ("Left", TOTAL_RIGHT),
        ("Right", TOTAL_LEFT),
        ("STATUS", STATUS),
    ]

    for (i, (k, v)) in enumerate(info):
        TEXT = "{}: {}".format(k, v)
        cv2.putText(frame, TEXT, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


    TOTAL_FRAMES += 1
    fps.update()

  fps.stop()
  print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
  print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))'''




# will not be necessary in final code as final code uses boundary around door instead of a line
#import cv2
#def output(frame):
#output the video withfor now a centre line ,the ids, wheter it is waiting,tracking or dectecting people ,and
# the count of left and right
       # draw a horizontal * vertical -- line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'Right' or 'Left'  ** up or down
 #   W = None
  #  H = None
    #if W is None or H is None:
     # (H, W) = frame.shape[:2]

    #line = cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 0, 255), 2)


    #print(line)
    #return line
#import numpy as np
#import cv2

#def mess(video):
 #   font = cv2.FONT_HERSHEY_SIMPLEX
  #  cv2.putText(frame,'hello',(10,500), font, 1,(255,255,255),2)
