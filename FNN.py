from copy import deepcopy
from utils.liveupdate import *
from utils.storage import *
from dataparser import *
from activator import *
import numpy as np
import filters
import pickle
import signal
import time
import os

class Neuron:
	def __init__(self, synapse=None, layerActivation=None, bias=None, activator=sigmoid):
		self.next = None
		self.back = None
	
		self.layerActivation = layerActivation
		self.layerPlain = None
		self.synapse = synapse
		self.bias = bias
		self.activator = activator	
	
		self.error = None
		self.number = 0

class NeuronNetwork:
	def __init__(self, trainingInput, trainingOutput):
		self.trainInput = trainingInput
		self.trainOutput = trainingOutput

		self.firstNeuron = Neuron(layerActivation=self.trainInput)
		self.lastNeuron = self.firstNeuron	

	def createNeuron(self, layerDimensions, activator=sigmoid, setAxis=False):
		bias = np.random.randn(layerDimensions[1])
		synapse = np.random.rand(layerDimensions[0], layerDimensions[1])
		
		newNeuron = Neuron(synapse=synapse, bias=bias, activator=activator)
		newNeuron.number = self.lastNeuron.number + 1
		newNeuron.setAxis = setAxis

		self.connectNeurons(self.lastNeuron, newNeuron)
		self.lastNeuron = self.lastNeuron.next

	def connectNeurons(self, backNeuron, nextNeuron):
		backNeuron.next = nextNeuron
		nextNeuron.back = backNeuron

		def isNoneType(value):
			return (type(value) == type(None))
	
		if(not isNoneType(backNeuron.synapse)):
			if not (backNeuron.synapse.shape[-1] == nextNeuron.synapse.shape[0]):
				raise Exception("Bad neuron arcitecture {}	->	{}".format(backNeuron.number, nextNeuron.number))
		elif not(backNeuron.layerActivation.shape[-1] == nextNeuron.synapse.shape[0]):
			raise Exception("Bad neuron arcitecture {}	->	{}".format(backNeuron.number, nextNeuron.number))


	def predict(self, trainingInput):
		currentNeuron = self.firstNeuron
		feedForward = trainingInput
		while currentNeuron.next != None:			
			feedForward = currentNeuron.activator(np.dot(feedForward, currentNeuron.next.synapse))
			currentNeuron = currentNeuron.next
		return currentNeuron.activator(feedForward)

	def classifyImage(self, inputImage):
		imageConvilution = self.data.parsed(inputImage)
		classification = np.zeros((self.lastNeuron.synapse.shape[-1],))
		for i in range(convolutionLayer.kernelCount):
			response = self.predict(np.hstack(imageConvilution[i, :, :]))
			classification += response
		classification = classification / convolutionLayer.kernelCount
		return np.argmax(classification)

	def testNetwork(self, testingInput, testingOutput):
		error = 0
		for testimage in range(testingInput.shape[0]):
			if not np.argmax(testingOutput[testimage]) == self.classifyImage(testingInput[testimage]):
				error += 1
		print("Error rate {}".format(error/testingInput.shape[0] * 100))



if __name__ == "__main__":
	convolutionLayer = filters.Convilution()
	convolutionLayer.addKernel(np.array([[[-1, 0, 1],   
										[-1, 0, 1],   
										[-1, 0, 1]]]))
	convolutionLayer.addKernel(np.array([[[0, 1, 0],   
										[1, -4, 1],   
										[0, 1, 0]]]))

	convolutionLayer.addKernel(np.array([[[-1,-1,-1],
										 [0,0,0],
										 [1,1,1]]]))	

	data =  DataParser(convolutionLayer)
	trainingInput, trainingOutput, testingInput, testingOutput = data.trainingInput, data.trainingOutput, data.testingInput , data.testingOutput

	neuralNetwork = None
	errorPlot = livePlot()			
	
	if(neuralNetwork == None):
		neuralNetwork = NeuronNetwork(trainingInput, trainingOutput)
		neuralNetwork.createNeuron((trainingInput.shape[-1], 4), activator=sigmoid)
		neuralNetwork.createNeuron((4, trainingOutput.shape[-1]), activator=specialSoftmax, setAxis=True)
		neuralNetwork.data = data
	for count in range(100000):
		#	skipping input neuron
		currentNeuron =  neuralNetwork.firstNeuron.next	

		# 	feedforward
		while currentNeuron != None:
			currentNeuron.layerPlain = currentNeuron.back.layerActivation.dot(currentNeuron.synapse)	+ currentNeuron.bias	
			currentNeuron.layerActivation = currentNeuron.activator(currentNeuron.layerPlain, currentNeuron.setAxis)	
			currentNeuron = currentNeuron.next

		currentNeuron = neuralNetwork.lastNeuron

		if(count % 100 == 0):
			errorPlot.add(np.sum(-neuralNetwork.trainOutput * np.log((currentNeuron.layerActivation))))
			
		#	backpropagation
		while currentNeuron.back != None:
			currentNeuron.newSyn = deepcopy(currentNeuron.synapse)
			if(currentNeuron.next == None):
				currentNeuron.error = (currentNeuron.layerActivation - neuralNetwork.trainOutput)
				currentNeuron.bias -= (10e-6) * currentNeuron.error.sum(axis=0)	
				currentNeuron.newSyn -= (10e-6) * currentNeuron.back.layerActivation.T.dot(currentNeuron.error)
			else:
				currentNeuron.error = (currentNeuron.next.error).dot(currentNeuron.next.synapse.T)
					
				delta = currentNeuron.error * currentNeuron.activator(currentNeuron.layerPlain, deriv=True)
		
				currentNeuron.bias -= (10e-6) * delta.sum(axis=0)	
				currentNeuron.newSyn -= (10e-6) * currentNeuron.back.layerActivation.T.dot(delta)
				#	can now update the layer in front. 
				currentNeuron.next.synapse = currentNeuron.next.newSyn			
			currentNeuron = currentNeuron.back
		currentNeuron.next.synapse = currentNeuron.next.newSyn				


	neuralNetwork.testNetwork(testingInput, testingOutput)
