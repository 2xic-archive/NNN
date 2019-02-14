import os
import sys
import pickle

def getpath():
	return "/".join(os.path.realpath(__file__).split("/")[:-2]) + "/"

def loadState(i):
	location = "{}state/{}.pkl".format(getpath(), i)
	print("Loading ... {}".format(location))
	if(os.path.exists(location)):
		file = open(location, "rb")
		response = pickle.load(file)
		file.close()
		print("Loaded the state")
		return response
	else:
		raise Exception("File not found {}".format(location))

def saveState(inputclass, i):
	location = "{}state/{}.{}".format(getpath(), i, "pkl")
	print("Saving ... {}".format(location))
	output = open(location, 'wb')
	pickle.dump(inputclass, output)
	output.close()
