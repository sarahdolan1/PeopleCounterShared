import argparse
import cv2
import numpy as np

def main():
	#passing arguments
	args = parse_arguments()
	#filepath to the video
	filepath =  "videos/webcamfootage.mp4"
	image_filepath ="ppl.jpg"
	#load the video
	# video = cv2.VideoCapture(filepath)
	#initialize the network only once
	net = load_network(args["prototxt"], args["model"])
	#iterating through the video
	#while video.isOpened():
	#	ret, frame = video.read()
	#	people_list = get_people(frame,net)
	#	break
	frame = cv2.imread(image_filepath)
	people_list = get_people(frame,net)
	image = display_output(frame, people_list)
	cv2.imshow("image", image)
	cv2.waitKey()

def display_output(image, res, min_confidence=0.4):
	"""
	Decode and display the output.
	Args:
		image: Input image.
		res: Network result.
	Returns:
		image: Image with boxes drawn on it.
	"""
	# Initialize boxes and classes.
	boxes, classes = {}, {}
	data = res[0][0]
	print(data.shape)
	print(data[5])
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
	# Draw boxes for all predictions.
	for imid in classes:
		for box in boxes[imid]:
			cv2.rectangle(image, (box[0], box[1]),
							(box[2], box[3]), (232, 35, 244), 2)
	# Return image with boxes drawn on it.
	return image

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
	ap.add_argument(
		"-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file"
	)
	ap.add_argument(
		"-m", "--model", required=True, help="path to Caffe pre-trained model"
	)
	ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
	ap.add_argument(
		"-o", "--output", type=str, help="path to optional output video file"
	)
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

def get_people(frame, net, min_confidence=0.4):
	""" detect people

	Args:
		frame (uint8): image
		net (cv2.dnn): neural network
		min_confidence (float): minimum for detection, defaults to 0.4

	Returns:
		people_list: list of bounding boxes containing people
	"""

	# initialize the list of class labels MobileNet SSD was trained to detect

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
        "person", #15
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

	people_list = []
	(height, width) = frame.shape[:2]

	# convert the frame to a blob and pass the blob through the network and
	# obtain the detections
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
	net.setInput(blob)
	detections = net.forward()

	return detections

def update_tracker(ct, people_list, frame, net, skip_frames=40):
	"""Track people over time

	Args:
		ct ([type]): dlib centroid tracker
		people_list ([type]): list of bounding boxes containing people
		frame ([type]): image
		net ([type]): neural network

	Returns:
		None
	"""

def write_text(img, text, pos, bg_color):
	"""[summary]

	Args:
		img ([type]): image
		text ([type]): overlay text
		pos ([type]): position of the text
		bg_color ([type]): text colour
	"""

# def output():

if __name__ == "__main__":
    main()
