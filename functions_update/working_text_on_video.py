import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
webcamfootage = "videos/webcamfootage.mp4"
cap = cv2.VideoCapture(webcamfootage)

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

def __draw_label(img, text, pos, bg_color):
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

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    
   # draw the label into the frame
   __draw_label(frame, 'Hello World', (20,20), (255,0,0))

   # Display the resulting frame
   cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
   if cv2.waitKey(1) & 0xFF == ord('q'):
     break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
