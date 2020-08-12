# People Counter Shared
As part of our summer internship, Sorcha and I have been asked to make a people counter code.
This code uses openCV and python to track people walking across a line in the middle of the screen whilst tracking their direction.
inspiration for the code https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

## Recent Updates:
* All necessary documents have been uploaded; (videos have not, as no videos will be shared for this code).
* We are currently adding functions to code so there are constant changes in this Git.

## How To Use:
for use run : python new_main.py -p mobilenet_ssd\MobileNetSSD_deploy.prototxt -m mobilenet_ssd\MobileNetSSD_deploy.caffemodel  -i videos/example_03.mp4 -o output/output_03.avi

## Folders
mobilenet_ssd => Contains the MobileNet SSD model used (prototxt and caffe files)  
example_inputs => Contains an example video to use for the input argument  
example_outputs => Contains the outputs given by the example input video  
pyimagesearch => Contains further code used to support the main people counter files

## Arguments
-p and -m provide the model. These are required and can be typed exactly as shown.  
-i and -o are optional, input & output arguments.  

For the input, the example input LeftRight.mp4 can be used or any video you choose.  
If no input is given, the live video from your webcam will be used instead.  
For the output, the examples show the output from running with the example input.  
When running any input video file or live video, the output can be saved and stored using the output argument, with whatever you wish to call the file.  
If no output argument is given, the output shown will not be saved or stored.

## Setup
     1. Make sure you have OpenVINO installed
     2. Install the required libraries (OpenCV, NumPy, SciPy, dlib, imutils) by searching online or using pip install
     3. Download this git repo as a zip
     4. Extract the file to the Documents directory

## Running The Code
     5. Change directory
          ~ cd C:\Users\<USERNAME>\Documents\PeopleCounterShared
     6. Run the code
          ~ python new_code.py -p mobilenet_ssd\MobileNetSSD_deploy.prototxt -m mobilenet_ssd\MobileNetSSD_deploy.caffemodel -i example_inputs\LeftRight.mp4 -o example_outputs\LeftRight-O.avi
