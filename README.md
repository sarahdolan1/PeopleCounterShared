# PeopleCounterShared
As part of our summer internship myself and sorcha have been asked to make a people counter code.
This code uses openCV and python to track people walking across a line in the middle of the screen whislt tracking their direction.
inpsiration for the code https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

# how to use :
for video use run : python people_counter.py -p mobilenet_ssd\MobileNetSSD_deploy.prototxt -m mobilenet_ssd\MobileNetSSD_deploy.caffemodel  -i videos/example_03.mp4 -o output/output_03.avi

for live video cam use run: python people_counter.py -p mobilenet_ssd\MobileNetSSD_deploy.prototxt -m mobilenet_ssd\MobileNetSSD_deploy.caffemodel
