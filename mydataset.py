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

def getImagesPathsFromLabelsFile():
	posesFile = '../Cropped/024_poseonly_normalised180.txt'
	listPaths = []
	
	with open(posesFile,'r') as f:
			i=0
			for line in f:
				path = line.split(' ')[0]
				if(path.split('/')[0]=='020'):
					path = os.path.join("../Cropped/GeneratedImgs",path)
					#path = ' '.join((line.split(' ')[1],line.split(' ')[2],line.split(' ')[3][0:len(line.split(' ')[3])-2],path))
					listPaths.append(path)
				elif(path.split('/')[0]=='022'):
					path = os.path.join("../Cropped/GeneratedImgs",path)
					#path = ' '.join((line.split(' ')[1],line.split(' ')[2],line.split(' ')[3][0:len(line.split(' ')[3])-2],path))
					listPaths.append(path)
				elif(path.split('/')[0]=='014b'):
					path = os.path.join("../Cropped/img_014b/GeneratedImgs",path)
					#path = ' '.join((line.split(' ')[1],line.split(' ')[2],line.split(' ')[3][0:len(line.split(' ')[3])-2], path))
					listPaths.append(path)
				print(i)
				i=i+1
				
	return listPaths
	
	


if __name__=='__main__':
	print('Getting all the images paths...')
	listPaths = getImagesPathsFromLabelsFile()
	random.seed(123)
	print('Shuffling the list')
	random.shuffle(listPaths)
	nbTrainingSet = int(math.floor(len(listPaths)*90/100))
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

	
