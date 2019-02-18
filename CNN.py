import matplotlib.pyplot as plt
from utils.liveupdate import *
from utils.storage import *
from copy import deepcopy
from dataparser import * 
from activator import *
from filters import * 
import numpy as np
import dataparser
import pickle
import os

def isNoneType(value):
	return (type(value) == type(None))

class Neuron:
	def __init__(self, weights, number, neuronType, activator):		
		self.next = None
		self.back = None

		if not isNoneType(weights):
			self.weights = weights
			self.bias = np.zeros((self.weights.shape[0],1))
		
			self.miniBatchWeigths =  np.zeros(self.weights.shape)
			self.miniBatchlBias =  np.zeros(self.bias.shape)

			self.orginalWeights = deepcopy(self.weights)
			self.orginalBias = deepcopy(self.bias)

		self.neuronType = neuronType
		self.activator = activator
		self.number = number

class Network:
	def __init__(self, weights=None, errorPlot=None):
		self.batchingSize = 64
		self.errorPlot = errorPlot

		self.startNeuon = None
		self.lastNeuron = None
		self.neuron = None
		self.names = {

		}

	def connect(self, weights, name="", layerType=None, activator=None):
		if(isNoneType(self.startNeuon)):
			self.startNeuon = Neuron(weights, 0, layerType, activator)
			self.neuron = self.startNeuon	
		else:
			self.neuron.next = Neuron(weights, self.neuron.number+1, layerType, activator)
			self.neuron.next.back = self.neuron
			self.neuron = self.neuron.next
			self.lastNeuron = self.neuron

		if(len(name) > 0):
			self.neuron.name = name
		else:
			self.neuron.name = "id:{}".format(self.neuron.number)
		self.names[name] = self.neuron
				
	def feedForward(self, data=None, training=True):
		if(self.neuron.neuronType == "maxpool"):
			self.neuron.convData = Convilution.poolingCNN(data)
			#	flatten
			zSize = self.neuron.convData.shape[0]
			ySize = self.neuron.convData.shape[1]
			self.neuron.activation = self.neuron.convData.reshape((zSize * ySize * ySize, 1))
		elif(self.neuron.neuronType == "convolution"):
			self.neuron.activation = Convilution.convilutionCNN(data, self.neuron.weights, self.neuron.bias)
		else:
			self.neuron.activation = self.neuron.weights.dot(data) + self.neuron.bias
		

		if(self.neuron.activator == "relu"):
			self.neuron.activation[self.neuron.activation<=0] = 0
		elif(self.neuron.activator == "softmax"):
			self.neuron.activation = softmax(self.neuron.activation)

			if(self.neuron.number == self.lastNeuron.number and training):
				self.neuron.delta = (self.neuron.activation - self.output)

			return self.neuron.activation

		if self.neuron.number < self.lastNeuron.number:
			self.next()
			return self.feedForward(self.neuron.back.activation)


	def feedBackward(self, data=None, custom=None):	
		if(not isNoneType(self.neuron.neuronType)):
			if(self.neuron.neuronType == "convolution"):
				self.neuron.activation, self.neuron.weights, self.neuron.bias = Convilution.convilutionBackwardCNN(self.neuron.next.activation,	data, self.neuron.weights)				
			elif(self.neuron.neuronType == "maxpool"):
				if(self.neuron.name == "flatten"):
					self.neuron.convData = self.neuron.delta.reshape(self.neuron.convData.shape)
				self.neuron.activation = Convilution.poolingBackwardCNN(self.neuron.convData, self.neuron.back.activation)
			if(not isNoneType(self.neuron.back) and self.neuron.back.activator == "relu"):
				self.neuron.activation[self.neuron.back.activation<=0] = 0
		else:
			previousDelta = self.neuron.delta

			self.neuron.delta = self.neuron.weights.T.dot(previousDelta)
			self.neuron.weights = previousDelta.dot(self.neuron.back.activation.T)
			self.neuron.bias = previousDelta.sum(axis = 1).reshape(self.neuron.bias.shape)

			if(self.neuron.back.activator == "relu"):
				self.neuron.delta[self.neuron.back.activation<=0] = 0
			self.neuron.back.delta = self.neuron.delta	

		if(self.neuron.number > 1):
			self.back()
			return self.feedBackward(self.neuron.back.activation)
		elif((self.neuron.number-1) == 0):
			self.back()
			return self.feedBackward(self.input)

	def updateBatchweigths(self):
		while self.neuron != None:
			if(self.neuron.name == "flatten"):
				self.next()
				continue

			#	update from mini-batch
			self.neuron.miniBatchWeigths += self.neuron.weights
			self.neuron.miniBatchlBias += self.neuron.bias
	
			self.neuron.weights = self.neuron.orginalWeights
			self.neuron.bias = self.neuron.orginalBias
		
			self.next(protection=False)
		self.getFrontNeuron()

	def graidientDescent(self):
		def ADAM(g, target, beta1=0.95, beta2=0.99, learningRate=0.01):
			#	http://ruder.io/optimizing-gradient-descent/index.html#adam
			m = np.zeros((g.shape))
			v = np.zeros((g.shape))

			m = beta1 * m + (1-beta1) * g 
			v = beta2 * v + (1-beta2) * (g**2)
	
			target -= learningRate * m/np.sqrt(v+1e-8)
		ADAM((self.neuron.miniBatchlBias/self.batchingSize), target=self.neuron.bias)
		ADAM((self.neuron.miniBatchWeigths/self.batchingSize), target=self.neuron.weights)

	def prepareForBatch(self):
		while self.neuron != None:
			if(self.neuron.name == "flatten"):
				self.back()
				continue

			self.neuron.miniBatchWeigths =  np.zeros(self.neuron.weights.shape)
			self.neuron.miniBatchlBias =  np.zeros(self.neuron.bias.shape)
			
			self.back(protection=False)
		self.getFrontNeuron()


	def optimize(self):
		while self.neuron != None:
			if(self.neuron.name == "flatten"):
				self.next()
				continue
			self.graidientDescent()
			self.next(protection=False)
		self.getLastNeuron()
		self.prepareForBatch()

	def miniBatch(self, X, Y):
		for index in range(self.batchingSize):
			self.input = X[index]
			self.output = Y[index]
			self.feedForward(self.input)
			self.feedBackward()

			loss = np.sum(-self.output * np.log((self.lastNeuron.activation)))
			self.errorPlot.add(loss)	

			self.updateBatchweigths()
		self.optimize()


	def back(self, protection=True):
		if (not isNoneType(self.neuron.back) or not protection):
			self.neuron = self.neuron.back

	def next(self, protection=True):
		if (not isNoneType(self.neuron.next) or not protection):
			self.neuron = self.neuron.next

	def getFrontNeuron(self):
		self.neuron = self.startNeuon

	def getLastNeuron(self):
		self.neuron = self.lastNeuron

	def predict(self, image):
		self.getFrontNeuron()
		probs = self.feedForward(image, training=False)
		return np.argmax(probs)

	def filter(self, size):
		return np.random.normal(loc= 0, scale=(1/np.sqrt(np.prod(size))), size=size)

	def weight(self, size):
		return np.random.rand(size[0], size[1]) 

	def testNetwork(self, testingInput, testingOutput):
		error = 0
		for testImage in range(testingInput.shape[0]):
			inputImage = np.zeros((1, 28, 28))
			inputImage[0, : , : ] = testingInput[testImage]
			self.input = inputImage

			if not self.predict(self.input) == np.argmax(testingOutput[testImage]):
				error += 1

		print("Error == {}".format(error/testingInput.shape[0] * 100))

if __name__ == "__main__":
	data = DataParser()
	trainingInput, trainingOutput, testingInput, testingOutput = data.trainingInput, data.trainingOutput, data.testingInput , data.testingOutput

	neuralNetwork = Network(errorPlot=livePlot())
	neuralNetwork.connect(neuralNetwork.filter((8,1,5,5)), name="f1", layerType="convolution", activator="relu")
	neuralNetwork.connect(neuralNetwork.filter((8,8,5,5)), name="f2", layerType="convolution", activator="relu")
	neuralNetwork.connect(None 							, name="flatten", layerType="maxpool")
	neuralNetwork.connect(neuralNetwork.weight((128,800)), name="w3", activator="relu")
	neuralNetwork.connect(neuralNetwork.weight((trainingOutput.shape[1], 128)), name="w4", activator="softmax")
	neuralNetwork.getFrontNeuron()

	for epochs in range(3):
		for count in range(0, trainingInput.shape[0], neuralNetwork.batchingSize):
			batchingX = np.zeros((neuralNetwork.batchingSize, 1, trainingInput.shape[-2], trainingInput.shape[-1]))
			batchingY = np.zeros((neuralNetwork.batchingSize, trainingOutput.shape[1], 1))

			for x in range(neuralNetwork.batchingSize):
				if(x > trainingInput.shape[0]):
					batchingX[x][0] = trainingInput[count + x,:]
					batchingY[x][np.argmax(trainingOutput[count + x])] = 1
				else:
					batchingX[x][0] = trainingInput[x,:]
					batchingY[x][np.argmax(trainingOutput[x])] = 1

			neuralNetwork.getFrontNeuron()
			neuralNetwork.miniBatch(batchingX, batchingY)
		neuralNetwork.testNetwork(testingInput, testingOutput)



