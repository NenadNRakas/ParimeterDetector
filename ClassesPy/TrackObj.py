class TrackableObject:
	def __init__(self, objectID, centroid):
		# Store an ID, Initialize a CentroID List with present CentroID
		self.objectID = objectID
		self.centroids = [centroid]
 
		# Object counter check, initialize a boolean
		self.counted = False
