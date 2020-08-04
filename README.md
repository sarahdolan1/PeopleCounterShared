# People Counter Shared
As part of our summer internship myself and sorcha have been asked to make a people counter code.
This code uses openCV and python to track people walking across a line in the middle of the screen whislt tracking their direction.
inpsiration for the code https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

# Recent Updates:
* All neccessary documents have been uploaded; (videos have not, as no videos will be shared for this code).
* We are currently adding functions to code so there is constant changes in this Git.
* Function_updates is the file that contains our "functioned" code. please ingore that file.

# How To Use:
for video use run : python people_counter.py -p mobilenet_ssd\MobileNetSSD_deploy.prototxt -m mobilenet_ssd\MobileNetSSD_deploy.caffemodel  -i videos/example_03.mp4 -o output/output_03.avi

for live video cam use run: python people_counter.py -p mobilenet_ssd\MobileNetSSD_deploy.prototxt -m mobilenet_ssd\MobileNetSSD_deploy.caffemodel


## Folders
mobilenet_ssd => Contains the MobileNet SSD model used (prototxt and caffe files)  
example_inputs => Contains an example video to use for the input argument  
example_outputs => Contains the outputs given by the example input video  
pyimagesearch => Contains further code used to support the main people counter files

## Arguments
-p and -m provide the model. These are required and can be typed exactly as shown.  
-i and -o are optional, input and output arguments.  

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

## Running the code
     5. Change directory
          ~ cd C:\Users\<USERNAME>\Documents\PeopleCounterShared
     6. Run the code
          ~ python people_counter.py -p mobilenet_ssd\MobileNetSSD_deploy.prototxt -m mobilenet_ssd\MobileNetSSD_deploy.caffemodel -i example_inputs\LeftRight.mp4 -o example_outputs\LeftRight-O.avi
