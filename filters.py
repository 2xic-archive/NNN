import numpy as np
import matplotlib
import matplotlib.pyplot as plt
	
class Convilution:
	def __init__(self):
		self.kernelList = np.zeros((3,3,3))  
		self.kernelCount = 0

	def addKernel(self, kernel):
		listSize = self.kernelList.shape[0]
		if(listSize < (self.kernelCount + 1)):
			newKernelList = np.zeros((listSize * 2,3,3))  
			for index in range(listSize):
				newKernelList[index, :, :] = self.kernelList[index]
			self.kernelList = newKernelList
		self.kernelList[self.kernelCount, :, :] = kernel
		self.kernelCount += 1

	def conviulutionKernel(self, kernelList, inputImage):
		kernelSizeXY = 3
		results = np.zeros((inputImage.shape))
		for yPosition in range(inputImage.shape[0]-kernelSizeXY):
			for xPosition in range(inputImage.shape[1]-kernelSizeXY):
				windowArea = inputImage[yPosition:yPosition+kernelSizeXY,
									xPosition:xPosition+kernelSizeXY]
				appliedKernel = windowArea * kernelList
				results[yPosition, xPosition] = np.sum(appliedKernel)
		return results


	def reluKernel(self, appliedKernel):
		reluLayer = np.zeros(appliedKernel.shape)  		
		for kernelNumber in range(appliedKernel.shape[0]):  
			for yPosition in range(appliedKernel.shape[1]):  
				for xPosition in range(appliedKernel.shape[2]):  
					reluLayer[kernelNumber, yPosition, xPosition] = np.max([appliedKernel[kernelNumber, yPosition, xPosition], 0]) 
		return reluLayer

	def poolKernel(self, appliedKernel, size=2, stride=5):
		poolingLayer = np.zeros((
							appliedKernel.shape[0],
							int( 1+(appliedKernel.shape[1]-size)/stride),  
							int( 1+(appliedKernel.shape[2]-size)/stride),  
						)) 
		for kernelNumber in range(appliedKernel.shape[0]):  
			stepY = 0
			for yPosition in range(0, appliedKernel.shape[1]-size-1, stride):  
				stepX = 0
				for xPosition in range(0, appliedKernel.shape[2]-size-1, stride):  
					poolingLayer[kernelNumber, stepY, stepX] = np.max([appliedKernel[kernelNumber, yPosition:yPosition+size,  xPosition:xPosition+size]]) 
					stepX += 1
				stepY += 1			
		return poolingLayer  

	def applyKernels(self, inputImage):
		appliedKernel = np.zeros((
									self.kernelCount,
									inputImage.shape[0],
									inputImage.shape[1],
								))
		for i in range(self.kernelCount):
			appliedKernel[i, :, :] = self.conviulutionKernel(self.kernelList[i, :], inputImage)
		return appliedKernel


	def convilutionCNN(inputImage, filterInput, bias, stride=1):
		filterXY = filterInput.shape[2]     
		maxY = inputImage.shape[1] - filterXY + 1
		maxX = inputImage.shape[2] - filterXY + 1		

		strideShape = int((inputImage.shape[1] - filterXY)/stride) + 1
		results = np.zeros((
			filterInput.shape[0],
			strideShape,
			strideShape
		))
		for currentFilter in range(filterInput.shape[0]):
			for xPosition in range(0, maxY, stride):
				for yPosition in range(0, maxX, stride):
					windowArea = inputImage[:,
											xPosition:xPosition+filterXY,
											yPosition:yPosition+filterXY]
					results[currentFilter, xPosition, yPosition] = np.sum(windowArea * filterInput[currentFilter]) + bias[currentFilter]
		return results
	

	def poolingCNN(inputImage, size=2, stride=2):
		results = np.zeros((
								inputImage.shape[0],  
								int(1 + (inputImage.shape[1] - size) / stride),  
								int(1 + (inputImage.shape[2] - size) / stride)
							))
		for filterID in range(inputImage.shape[0]):  
			stepY = 0
			for positionY in range(0,inputImage.shape[1], stride):  
				stepX = 0
				for positionX in range(0, inputImage.shape[2], stride):  
					results[filterID, stepY, stepX] = np.max([inputImage[filterID, positionY:positionY+size,  positionX:positionX+size]]) 
					stepX += 1
				stepY += 1			
		return results

		
	def convilutionBackwardCNN(lastBack, orginalInput, orginalFilter, stride=1):
		resultsInput = np.zeros((
			orginalInput.shape
		))
		resultsFilt = np.zeros((
			orginalFilter.shape
		))
		resultsBias = np.zeros((
			orginalFilter.shape[0], 
			1
		))

		filterXY = orginalFilter.shape[2]	
		strideShape = ((orginalInput.shape[1] - filterXY) + 1) 
			
		for currentFilter in range(orginalFilter.shape[0]):
			stepX = 0
			for xPadding in range(0, strideShape, stride):
				stepY = 0
				for yPadding in range(0, strideShape, stride):
					region = orginalInput[:,
										xPadding:xPadding+filterXY,
										yPadding:yPadding+filterXY]
					resultsFilt[currentFilter] += lastBack[currentFilter, stepX, stepY] * region
					resultsInput[:, xPadding:xPadding+filterXY, yPadding:yPadding+filterXY] += lastBack[currentFilter, stepX, stepY] * orginalFilter[currentFilter]
					stepY += 1
				stepX += 1
				resultsBias[currentFilter] = np.sum(lastBack[currentFilter])
		return resultsInput, resultsFilt, resultsBias
			
	def poolingBackwardCNN(orginalPool, activaton, size=2, stride=2):
		results = np.zeros(activaton.shape)
		for filterID in range(activaton.shape[0]):
			stepY = 0
			shape = (activaton.shape[1]-size+1)
			for positionY in range(0, shape, stride):
				stepX = 0
				for positionX in range(0, shape, stride):				
					windowArea = activaton[filterID, positionY:positionY+size, positionX:positionX+size]
					XmaxIndex, YmaxIndex = np.where(windowArea == windowArea.max())
					results[filterID, positionY+XmaxIndex[0], positionX+YmaxIndex[0]] = orginalPool[filterID, stepY, stepX]
					stepX += 1
				stepY += 1		
		return results


def preview(image):
	convilution = Convilution()
	convilution.addKernel(np.array([[[-1, 0, 1],   
							[-1, 0, 1],   
							[-1, 0, 1]]])	)
	convilution.addKernel(np.array([[[-1, -1, -1],   
							[-1, 8, -1],   
							[-1, -1, -1]]]))
	convilution.addKernel(np.array([[[0, 1, 0],   
							[1, -4, 1],   
							[0, 1, 0]]]))
	convilution.addKernel(np.array([[[1,0,-1],
							[2,0,-2],
							[1,0,-1]]]))	

	appliedKernel = convilution.applyKernels(image)
	poolingLayer = convilution.reluKernel(appliedKernel)
	reluLayer = convilution.poolKernel(appliedKernel)
	
	fig, ax = matplotlib.pyplot.subplots(nrows=3,ncols=convilution.kernelCount+1)  

	ax[0,0].imshow(image).set_cmap("gray")  
	ax[0,0].get_xaxis().set_ticks([])  
	ax[0,0].get_yaxis().set_ticks([])  
	ax[0,0].set_title("L1-Map2")  


	ax[1,0].get_xaxis().set_ticks([])  
	ax[1,0].get_yaxis().set_ticks([])  
	ax[1,0].spines["top"].set_visible(False)
	ax[1,0].spines["right"].set_visible(False)
	ax[1,0].spines["left"].set_visible(False)
	ax[1,0].spines["bottom"].set_visible(False)
	#	OK
	ax[2,0].get_xaxis().set_ticks([])  
	ax[2,0].get_yaxis().set_ticks([])  
	ax[2,0].spines["top"].set_visible(False)
	ax[2,0].spines["right"].set_visible(False)
	ax[2,0].spines["left"].set_visible(False)
	ax[2,0].spines["bottom"].set_visible(False)

	ax[0,0].imshow(image).set_cmap("gray")  
	ax[0,0].get_xaxis().set_ticks([])  
	ax[0,0].get_yaxis().set_ticks([])  
	ax[0,0].set_title("FILTER[0]")  

	for i in range(1, convilution.kernelCount+1):
		ax[0,i].imshow(appliedKernel[i-1, :, :]).set_cmap("gray")  
		ax[0,i].get_xaxis().set_ticks([])  
		ax[0,i].get_yaxis().set_ticks([])  
		ax[0,i].set_title("FILTER[{}]".format(i))  
	
		ax[1,i].imshow(reluLayer[i-1, :, :]).set_cmap("gray")  
		ax[1,i].get_xaxis().set_ticks([])  
		ax[1,i].get_yaxis().set_ticks([])  
		ax[1,i].set_title("RELU")  
		
		ax[2,i].imshow(poolingLayer[i-1, :, :]).set_cmap("gray")  
		ax[2,i].get_xaxis().set_ticks([])  
		ax[2,i].get_yaxis().set_ticks([])  
		ax[2,i].set_title("POOLING")  	
		
	matplotlib.pyplot.show(block=True)





