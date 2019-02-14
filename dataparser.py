import numpy as np
import filters
import numpy as np
import matplotlib.pyplot as plt
from utils.storage import * 

class DataParser:
	def __init__(self, filters=None):
		self.testImages = 100
		self.classes = 10
		self.total = 100 * self.classes 

		if(filters == None):
			self.trainingInput = np.zeros((self.total, 28, 28))			
			self.trainingOutput = np.zeros((self.total, self.classes))
			self.applyKernels = False
		else:
			self.trainingInput = np.zeros((self.total * filters.kernelCount, (28 * 28)))				
			self.trainingOutput = np.zeros((self.total * filters.kernelCount, self.classes))
			self.filters = filters
			self.applyKernels = True

		self.testingInput =  np.zeros((self.testImages, 28 , 28))
		self.testingOutput = np.zeros((self.testImages, self.classes))
		self.index = 0
		self.load()

	def normalize(self, inputImage):
		if((np.min(inputImage) != 0 or np.max(inputImage) != 0) and not (np.max(inputImage) == np.min(inputImage))):
			normalized = (inputImage-np.min(inputImage))/(np.max(inputImage)-np.min(inputImage))
			return normalized
		else:
			return inputImage

	def listLabel(self, listNumber, length):
		output = [0,] * length
		output[int(listNumber[0])] = 1
		return output

	def parsed(self, image):
		imageNormalized = self.normalize(image)				
		if(self.applyKernels):
			filterImages = self.filters.applyKernels(imageNormalized)
			return filterImages
		return imageNormalized

	def addImage(self, image, listNumber):
		if(self.applyKernels):
			filterImages = self.parsed(image)
			for filterid in range(0, self.filters.kernelCount):
				self.trainingInput[self.index, :] = np.hstack(filterImages[filterid, :, :])
				self.trainingOutput[self.index, :] = self.listLabel(listNumber, self.classes)
				self.index += 1
		else:
			self.trainingInput[self.index, :] = self.parsed(image)
			self.trainingOutput[self.index, :] = self.listLabel(listNumber, self.classes)
			self.index += 1			
		return self.index

	def load(self):
		train_data = np.loadtxt(getpath() + "dataset/mnist_train.csv", delimiter=",")
		fac = 750  * 0.99 + 0.01
		images = np.asfarray(train_data[:, 1:]) / fac
		labels = np.asfarray(train_data[:, :1])
		#	training
		for i in range(self.total):
			self.addImage(images[i].reshape((28,28)),
								labels[i].tolist())
		#	testing
		for i in range(self.total, self.total+self.testImages):
			self.testingInput[self.total - i, :] = images[i].reshape((28,28))
			self.testingOutput[self.total - i, :] = self.listLabel(labels[i].tolist(), self.classes)

		indices = np.arange(self.trainingInput.shape[0])
		np.random.shuffle(indices)
		self.trainingInput = self.trainingInput[indices]
		self.trainingOutput = self.trainingOutput[indices]
		