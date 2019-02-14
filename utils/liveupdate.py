import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time
import random

class livePlot:
	def __init__(self):
		self.X = []
		self.Y = []
 
		plt.ion()
		plt.xlim(0, 200) 
		self.graph = plt.plot(self.X,self.Y)[0]
		self.max = 0
		self.min = 0
 
	def add(self, Y_data):
		self.X.append(len(self.X)+1)
		self.Y.append(Y_data)
 
		if(Y_data > self.max):
			self.max = Y_data
			plt.ylim(self.min, self.max)
		
		if(Y_data < self.min):
			self.min = Y_data
			plt.ylim(self.min, self.max)
		  
		if(len(self.X) > 200):
			self.Y = self.Y[-100:]       
			self.X = list(range(len(self.Y)))
			self.min = min(self.Y)
			self.max = max(self.Y)
 
		self.graph.set_ydata(self.Y)
		self.graph.set_xdata(self.X)
 
 
		#   https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7
		def mypause(interval):
			backend = plt.rcParams['backend']
			if backend in matplotlib.rcsetup.interactive_bk:
				figManager = matplotlib._pylab_helpers.Gcf.get_active()
				if figManager is not None:
					canvas = figManager.canvas
					if canvas.figure.stale:
						canvas.draw()
					canvas.start_event_loop(interval)
					return
		mypause(0.02)
 
 
if __name__ == "__main__":
	start = livePlot()
	testData = [1000, 500, 300, -200, -100, -900, 800, 400, 300, -300]
	while True:
		start.add(testData[random.randint(0, len(testData)-1)] )

