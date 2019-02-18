import os
import sys
import pickle

def getpath():
	return "/".join(os.path.realpath(__file__).split("/")[:-2]) + "/"

def loadState(name):
	location = "{}state/{}.pkl".format(getpath(), name)
	print("Loading ... {}".format(location))
	if(os.path.exists(location)):
		file = open(location, "rb")
		response = pickle.load(file)
		file.close()
		print("Loaded the state")
		return response
	else:
		raise Exception("File not found {}".format(location))

def saveState(inputClass, name):
	location = "{}state/".format(getpath())
	if not (os.path.exists(location)):
		os.makedirs(location)
	location += "{}.{}".format(name, "pkl")
	print("Saving ... {}".format(location))
	output = open(location, 'wb')
	pickle.dump(inputClass, output)
	output.close()
