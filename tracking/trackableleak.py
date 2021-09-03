class TrackableLeak:
	def __init__(self, objectID, Leak):
		# store the object ID, then initialize a list of Leaks
		# using the current Leak
		self.objectID = objectID
		self.Leaks = [Leak]