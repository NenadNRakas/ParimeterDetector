# Necessary packages import
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=50):
		# Initialize new ID, two Dictionaries for ID, CentroID and continues object frames for lifecycle
		# Tag = "Disappeared"
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# Deregister Tracked Object after maximum disappointed frames
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# Register an object with next open ID and store the CentroID
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# Delete Object ID from both dictionaries to deregister an object
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# Check Input bounding box rect list is  Empty 
		if len(rects) == 0:
			# Mark disappeared tracked objects loop
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# Deregister the object after frame threshold
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# No tracking or CentroID info, Return
			return self.objects

		# In the current frame, initialize an array of input CentroID's
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# Bounding box rects loop
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# Derive the CentroID from bounding box coordinates
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# Not tracking objects, register each CentroID input
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# Tracking objects, map present and new CentroID Objects
		# centroids
		else:
			# Set a group of matching object ID's and CentroID's
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# Calculate present and input object CentroID distance for later mapping
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# Search and sort rows with lowest values at the front of index list, then match
			rows = D.min(axis=1).argsort()

			# Search, sort columns with lowest values based on computed row index list
			cols = D.argmin(axis=1)[rows]

			# Initialize Row and Column Index variables for tracking registration updates
			usedRows = set()
			usedCols = set()

			# Row, column index tuple loop
			for (row, col) in zip(rows, cols):
				# Ignore preprocessed row, column values
				if row in usedRows or col in usedCols:
					continue

				# Non processed object in present row, get the ID, set the new centroid, reset "Disapeared" variable
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# Add processed row, column index sets
				usedRows.add(row)
				usedCols.add(col)

			# Compute non processed row and column index
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# If CentroID's >= Input's, check "Disapeared" objects
			if D.shape[0] >= D.shape[1]:
				# Free row indexes loop
				for row in unusedRows:
					# Get Object ID in row index, add to "Disapeared" variable
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# Deregister object after "Disappeared" tag threshold in continues frames
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# Input's > CentroID's, register new CentroID Trackable Object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# Trackable object set returned
		return self.objects