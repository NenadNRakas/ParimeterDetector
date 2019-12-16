# Necessary packages
from ClassesPy.CenterIDTracking import CentroidTracker
from ClassesPy.TrackObj import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
	# Play an alarm audio
	playsound.playsound(path)

# Construct and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="Caffe deploy prototxt file, path")
ap.add_argument("-m", "--model", required=True, help="Caffe Pre-Trained model, path")
ap.add_argument("-i", "--input", type=str, help="Input video file, path")
ap.add_argument("-a", "--alarm", type=str, default="Sound\SubmarineDive.mp3", help="Audio file, path")
ap.add_argument("-o", "--output", type=str,	help="Output video file, path")
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="Weak detections filter, value")
ap.add_argument("-s", "--skip-frames", type=int, default=30, help="Skiped frames between detections, value")
args = vars(ap.parse_args())

# Initialize a class labels list trained with MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Serialized model loaded from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# No video argument, stream WebCam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	#vs = cv2.VideoCapture("")
	time.sleep(2.0)

# Or play the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])
	firstFrame = None
	#vs = cv2.VideoCapture("")
	time.sleep(2.0)

# Initialize the video writer variable
writer = None
 
# Initialize the image dimension variables
W = None
H = None
 
# Initialize a CentroID Tracker, a dlib trackers list, then a dictionary for object ID to TrackableObject mapping
# ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
ct = CentroidTracker()
trackers = []
trackableObjects = {}
 
# Initialize the variables
totalFrames = 0
totalDown = 0
totalUp = 0
ALARM_ON = False
 
# Start the FPS throughput estimation
fps = FPS().start()

# Video stream image loop
while True:
	# Reading from VideoCapture or VideoStream get and process the following image
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame
 
	# Empty frame in a video ends the process
	if args["input"] is not None and frame is None:
		break
 
	# Image resize for quicker processing - then convert BGR to RGB for dlib
	# frame = imutils.resize(frame, width=250)
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
	# Set empty dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
 
	# When recording, initialize a writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # Initialize status and bounding boxes list from Object Detector or Tracker
	status = "Scanning"
	rects = []
 
	# Tracker method check, if needed, run a demanding object detector in parallel
	if totalFrames % args["skip_frames"] == 0:
		# Set Status, then Initialize Object Trackers
		status = "Detecting"
		trackers = []
 
		# Detect by converting a frame to blob, then passing a blob through a network... Simple!
		# blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

        # Detection loop
		for i in np.arange(0, detections.shape[2]):
			# Get the prediction confidence
			confidence = detections[0, 0, i, 2]
 
			# Minimum confidence filter
			if confidence > args["confidence"]:
				# Get Class Label Index from the detection list
				idx = int(detections[0, 0, i, 1])
 
				# Set the class label to human
				if CLASSES[idx] != "person":
					continue

                # Compute object bounding box x and y coordinates
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
 
				# Start a dlib tracker correlation an object from bounding box coordinates
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)
 
				# Append a Tracker to Tracker List for jumping frames
				trackers.append(tracker)

    # Switch from object detection to tracking and save processing "BHP"
	else:
		# Trackers loop
		for tracker in trackers:
			# Change the system Status to Tracking 
			status = "Tracking"
 
			# Tracker update and store position's
			tracker.update(rgb)
			pos = tracker.get_position()
 
			# Object position
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
 
			# Adding the bounding box coordinates to the rectangle list
			rects.append((startX, startY, endX, endY))

    # Display a horizontal line to determine object movement and location (In, Out, Up, Down)
	#cv2.line(frame, (0, H // 3), (W, H // 5), (127, 255, 0), 1)
	#cv2.line(frame, (0, H // 2), (W, H // 2), (125, 255, 0), 1)
	cv2.line(frame, (0, H // 3), (W, H // 1), (125, 255, 0), 1)
	# Associate Old and New objects with CentroID Tracker
	objects = ct.update(rects)

    	# Tracked objects loop
	for (objectID, centroid) in objects.items():
		# Existing Trackable Object ID's check
		to = trackableObjects.get(objectID, None)
 
		# Create new trackable object ID
		if to is None:
			to = TrackableObject(objectID, centroid)
 
		# Utilize an existing ID
		else:
			# Object center difference of previous and current y-coord will determine direction
			# Negative = Up and Positive = Down
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)
 
			# Object detection counter and trigger
			if not to.counted:
				# Negative position below the line triger the alarm
				# Above the line is a positive position with no trigger just counter
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True
				# Positive position above the line = free zone
				# Negative Position below the line = trigger and count
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True
					# Audio on or off check and play
					if not ALARM_ON:
						ALARM_ON = True
						#TOTAL += 1
						# Audio file present check and play					
						if args["alarm"] != "":
							t = Thread(target=sound_alarm, args=(args["alarm"],))
							t.deamon = True
							t.start()
 
					# Display the detection
					cv2.putText(frame, "DETECTION ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (127, 0, 255), 2)
				else:
					#COUNTER = 0
					ALARM_ON = False
				# Display number of detection
				cv2.putText(frame, "Alert: {}".format(totalDown), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (127, 0, 255), 2)

		# Trackable Object variable
		trackableObjects[objectID] = to

    	# Display the ID and object center
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 255, 0), 1)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (127, 255, 0), -1)
 
	# Information Tuple
	info = [("In", totalDown), ("Out", totalUp), ("Status", status)]
 
	# Information tuple loop
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 0, 255), 1)

    	# Write image to disk check
	if writer is not None:
		writer.write(frame)
 
	# Display stream
	cv2.imshow("Human Boundary Detector:", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# Quit loop on key "q"
	if key == ord("q"):
		break
 
	# Add and display FPS
	totalFrames += 1
	fps.update()

# Stop and display FPS details
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# Release video writer check
if writer is not None:
	writer.release()
 
# Check for file input and stop stream
if not args.get("input", False):
	vs.stop()
 
# Otherwise, release the input video file pointer
else:
	vs.release()

# Close open frames
cv2.destroyAllWindows()
