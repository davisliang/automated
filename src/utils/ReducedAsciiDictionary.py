import numpy as np

class ReducedAsciiDictionary:
	""" Creates a dictionary that maps characters to an integer

	This is useful because not all characters are used and this
	method allows for low-dimensional character vectors.

	"""

	def __init__(self, dictionary, ranges):
		""" Initialize the dictionary based on a set of ranges

		The ranges parameter is a 2D numpy array, doubly inclusive.
		Recall that the function ord() gets the int value of a char.
		Recall that the function chr() gets the char value of an int.

		TO-DO: Add range array validity checker.

		"""

		self.dictionary = dictionary

		# default constructor runs if size is 0. 
		if(ranges.size == 0):
			self.ranges = np.array([[32,63],[96,127]])
		else: 
			self.ranges = ranges

		#build dictionary
		counter = 0
		numRanges = self.ranges.shape[0]
		for i in range(0,numRanges):
			start = self.ranges[i][0]
			end = self.ranges[i][1]
			for j in range(start,end+1):
				self.dictionary[chr(j)] = counter
				counter += 1
				
