import numpy as np
import random

def getBatch(data):
	if data=='training':
		pathsFile = 'trainingImagesPaths.txt'
	elif data == 'validation':
		pathsFile = 'validationImagesPaths.txt'
	else:
		raise Exception('Please choose training or validation as first argument')
	
	
	with open(pathsFile, 'r') as f:
    		listPaths = [line.rstrip('\n') for line in f]
		
	
	posesFile = '../Cropped/024_poseonly_normalised180.txt'

	open('labelsTraining.txt', 'a').close()
	ff = open('labelsTraining.txt','r+')
	for i in range(0,len(listPaths)):
		print(i)
		imagePath = listPaths[i]
		if(imagePath.split('/')[2]=='img_014b'):
			imageName = imagePath[34:]
		elif(imagePath.split('/')[2]=='GeneratedImgs'):
			imageName = imagePath[25:]
		else:
			raise Exception('Image not found')

		with open(posesFile,'r') as f:
			for line in f:
				if line.split(' ')[0]==imageName:
					yau = float(line.split(' ')[1])
					pitch = float(line.split(' ')[2])
					roll = float(line.split(' ')[3])
					ff.write(str(yau))
					ff.write(' ')
					ff.write(str(pitch))
					ff.write(' ')
					ff.write(str(roll))
					ff.write(' ')
					break	
			
		ff.write(imagePath)
		ff.write('\n')
	ff.close()
	return 1
	

if __name__ == '__main__':
	getBatch('training')
