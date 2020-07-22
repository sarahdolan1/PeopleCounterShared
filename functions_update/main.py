import argparse
import VIDEO_FUNCTION
import OUTPUT_FUNCTION
import os
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="path to file")
    args = parser.parse_args()
    video = VIDEO_FUNCTION.readVideo(args.filepath)

    while True:
        frame = VIDEO_FUNCTION.readFrame(video)
        VIDEO_FUNCTION.displayFrame(frame)
        #VIDEO_FUNCTION.exitFrame(frame)
        #OUTPUT_FUNCTION.output(frame)
       # OUTPUT_FUNCTION.output(line)
       # OUTPUT_FUNCTION.mess(frame)
      #  cv2.imshow("line",line)
       # print(frame)
        cv2.waitKey()
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
          break

if __name__ == "__main__":
   main()
