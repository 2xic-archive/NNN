import numpy as np

def sigmoid(x, derivation=False):
	if (derivation == True):
		f = sigmoid(x, derivation=False)
		return f*(1-f)
	return 1/(1 + np.exp(-x))

def softmax(x):
	output = np.exp(x - np.max(x))
	return output/np.sum(output)

def specialSoftmax(x, setAxis=False):
	if not setAxis:
		output = np.exp(x - np.max(x))
		return output/np.sum(output)
	else:
		output = np.exp(x - np.max(x))
		return output/np.sum(output, keepdims=True,axis=1)