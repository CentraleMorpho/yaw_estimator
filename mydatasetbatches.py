from scipy import ndimage
import numpy as np
import random

def getBatch(data, batch_size):
	if data=='training':
		pathsFile = 'trainingImagesPaths.txt'
	elif data == 'validation':
		pathsFile = 'validationImagesPaths.txt'
	else:
		raise Exception('Please choose training or validation as first argument')
	
	
	with open(pathsFile, 'r') as f:
    		listPaths = [line.rstrip('\n') for line in f]
		
	print(len(listPaths))
	image_batch_paths = random.sample(listPaths, batch_size)
	
	#Get the images matrix
	image_size = 39
	imagesMatrix = np.ndarray(shape=(batch_size, image_size, image_size),dtype=np.float32)
	i = 0
	for imagePath in image_batch_paths:
		try:
			image_data = (ndimage.imread(imagePath).astype(float))
		    	if image_data.shape != (image_size, image_size):
		        	raise Exception('Unexpected image shape')
			imagesMatrix[i, :, :] = image_data
		except IOError as e:
            		print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
		i=i+1


	#Get the labels matrix
	i = 0
	labelsMatrix = np.zeros([batch_size,3])
	posesFile = '../Cropped/024_poseonly_normalised180.txt'
	for imagePath in image_batch_paths:
		if(imagePath.split('/')[2]=='img_014b'):
			imageName = imagePath[34:]
		elif(imagePath.split('/')[2]=='GeneratedImgs'):
			imageName = imagePath[25:]
		else:
			raise Exception('Image not found')

		found = 0
		with open(posesFile,'r') as f:
			for line in f:
				if line.split(' ')[0]==imageName:
					yau = float(line.split(' ')[1])
					pitch = float(line.split(' ')[2])
					roll = float(line.split(' ')[3])
					labelsMatrix[i,0] = yau
					labelsMatrix[i,1] = pitch
					labelsMatrix[i,2] = roll
					found = 1
					break	
		#if(found==0):
		#	print(imageName)		

		i = i+1

	

	return imagesMatrix, labelsMatrix
	

if __name__ == '__main__':
	imagesMatrix, labelsMatrix = getBatch('validation', 10)
	print(imagesMatrix.shape)
	print(labelsMatrix)
	
		

