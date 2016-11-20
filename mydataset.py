import os
import random
import math

def getImagesPaths():
	listPaths = []
	for root, dirs, files in os.walk("../Cropped"):
	    for file in files:
		if file.endswith(".png"):
		     listPaths.append(os.path.join(root, file))
	return listPaths


if __name__=='__main__':
	print('Getting all the images paths...')
	listPaths = getImagesPaths()
	random.seed(123)
	print('Shuffling the list')
	random.shuffle(listPaths)
	nbTrainingSet = int(math.floor(len(listPaths)*70/100))
	print('Creating the training set')
	trainingSetPaths = listPaths[:nbTrainingSet]
	print('Creating the validation set')
	validationSetPaths = listPaths[nbTrainingSet+1:]
	
	print('Saving the training paths to a txt file')
	trainingSetPathsFile = open('trainingImagesPaths.txt','w')
	for item in trainingSetPaths:
		trainingSetPathsFile.write("%s\n" % item)
	trainingSetPathsFile.close()

	print('Saving the validation paths to a txt file')
	validationSetPathsFile = open('validationImagesPaths.txt','w')
	for item in validationSetPaths:
		validationSetPathsFile.write("%s\n" % item)
	validationSetPathsFile.close()
	print('Completed')

	
