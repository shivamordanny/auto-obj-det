import numpy as np
import cv2 as cv
import argparse
import sys
import os.path
import gc
import time
from tracking.leaktracker import LeakTracker
from tracking.trackableleak import TrackableLeak

# Initialize the parameters
confThreshold = 0.2  # Confidence threshold
nmsThreshold = 0.2  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image
totalFrames = 0 # Frame counter

# Construct argument parser and parse the arguments
ap = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
ap.add_argument('--image', help='Path to image file.')
ap.add_argument('--video', help='Path to video file.')
args = ap.parse_args()

# Load names of classes
classesFile = "v14.names"
classes = None
with open(classesFile, 'rt') as f:
	classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "tiny-yolov4-custom-v16.cfg"
modelWeights = "tiny-yolov4-custom-v16.weights"

# Load network
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# instantiate the Leak tracker, then initialize a list to store
# each of the dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableLeak
lt = LeakTracker(maxDisappeared=50, maxDistance=100)

# Get the names of the output layers
def getOutputsNames(net):
	# Get the names of all the layers in the network
	layers_names = net.getLayerNames()
	# Get the names of the output layers, i.e. the layers with unconnected outputs
	return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
	# Draw a bounding box.
	cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 1)
	label = '%.2f' % conf
	# Get the label for the class name and its confidence
	if classes:
		assert (classId < len(classes))
		label = '%s:%s' % (classes[classId], label)
	# Display the label at the top of the bounding box
	labelSize, baseLine = cv.getTextSize(classes[classId], cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	top = max(top, labelSize[1])
	cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
				 (255, 255, 255), cv.FILLED)
	cv.putText(frame, classes[classId], (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, rects = []):
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]
	# Scan through all the bounding boxes output from the network and keep only the
	# ones with high confidence scores. Assign the box's class label as the class with the highest score.
	classIds = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > confThreshold:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])
	# Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
	indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	for i in indices:
		i = i[0]
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
		if classIds[i] == 0:
			rects.append((left, top, left + width, top + height))
			# drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
	return rects


def unique_leaks(objects, trackableLeak = {}):
	for (objectID, Leak) in objects.items():
		# check to see if a trackable leak exists for the current
		# object ID
		tl = trackableLeak.get(objectID, None)

		# if there is no existing trackable leak, create one
		if tl is None:
			tl = TrackableLeak(objectID, Leak)
		# otherwise, there is a trackable leak so we can utilize it
		# to determine direction
		else:
			tl.Leaks.append(Leak)

		# store the trackable leak in the dictionary
		trackableLeak[objectID] = tl
		#print('trackable leak stored')

		# draw the bounding box for tracked objects
		text = "ID {}".format(objectID+1)
		cv.putText(frame, text, (Leak[0] - 10, Leak[1] - 10),
			cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv.circle(frame, (Leak[0], Leak[1]), 4, (0, 255, 0), -1)


def show_fps(img, fps):
	"""Draw fps number at top-left corner of the image."""
	font = cv.FONT_HERSHEY_PLAIN
	line = cv.LINE_AA
	fps_text = 'FPS: {:.2f}'.format(fps)
	cv.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
	cv.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
	return img


# Process inputs
winName = 'Leak detection'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "leak-det.avi"

if args.image:
	# Open the image file
	if not os.path.isfile(args.image):
		print("Input image file ", args.image, " doesn't exist")
		sys.exit(1)
	cap = cv.VideoCapture(args.image)
	outputFile = args.image[:-4] + '_leak.jpg'
elif args.video:
	# Open the video file
	if not os.path.isfile(args.video):
		print("Input video file ", args.video, " doesn't exist")
		sys.exit(1)
	cap = cv.VideoCapture(args.video)
	outputFile = args.video[:-4] + '_leak.avi'
else:
	# Webcam input
	cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if not args.image:
	vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
								(round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

fps = 0.0
tic = time.time()

while cv.waitKey(1) < 0:

	# get frame from the video
	hasFrame, frame = cap.read()

	# Stop the program if reached end of video
	if not hasFrame:
		print("Done processing !!!")
		# print("Output file is stored as ", outputFile)
		cv.waitKey(3000)
		gc.collect()
		# Release device
		cap.release()
		# close any open windows
		cv.destroyAllWindows()
		break

	# Create a 4D blob from a frame.
	blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

	# Sets the input to the network
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers
	outs = net.forward(getOutputsNames(net))

	# Remove the bounding boxes with low confidence
	rects = postprocess(frame, outs, rects = [])

	# use the Leak tracker to associate the (1) old object
	# Leaks with (2) the newly computed object Leaks
	objects = lt.update(rects)
	# Track Leaks and assign unique IDs
	unique_leaks(objects, trackableLeak = {})

	# Calculate FPS and print on the window
	img = show_fps(frame, fps)
	cv.imshow(winName, img)
	toc = time.time()
	curr_fps = 1.0 / (toc - tic)
	# calculate an exponentially decaying average of fps number
	fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
	tic = toc

	# Write the frame with the detection boxes
	if args.image:
		cv.imwrite(outputFile, frame.astype(np.uint8))
	else:
		vid_writer.write(frame.astype(np.uint8))

	totalFrames += 1
	# print("F R A M E: ",totalFrames)
