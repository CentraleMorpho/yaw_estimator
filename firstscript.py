###################################################
#PUT THIS IPYNB IN THE SAME DIRECTORY AS CROPPED
###################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
import pandas as pd




#Gets the images of the folder (normalized) as a 3D matrix 
def load_dataset(folder):
    image_size = 39  # Pixel width and height.
    pixel_depth = 255.0  # Number of levels per pixel.

    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - 
                            pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape')
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset
    

#Gets the matrix of the labels corresponding to the images on a folder file
def findLabelsMatrix(folder):
    #Get the full label file in a label dataframe
    labelsDf = pd.read_csv('../Cropped/024_poseonly_normalised180.txt', header=None)
    labelsDf.columns = ['string']
    imagenameFinder = lambda x: x.split(" ")[0]
    labelsDf['imagename']=labelsDf['string'].apply(imagenameFinder)
    
    #Get the name of the images in the folder and keep only these rows in the label dataframe
    image_files = os.listdir(folder)
    image_file_series = pd.Series(image_files)
    folderAdder = lambda x: folder.split("GeneratedImgs/")[1]+'/'+x
    image_file_series = image_file_series.apply(folderAdder)
    labelsDf = labelsDf.loc[labelsDf['imagename'].isin(image_file_series)]
    
    #Add to the label dataframe the 3 angles (by formatting the string)
    yauFinder = lambda x: x.split(" ")[1]
    labelsDf['Yau']=labelsDf['string'].apply(yauFinder)
    pitchFinder = lambda x: x.split(" ")[2]
    labelsDf['Pitch']=labelsDf['string'].apply(pitchFinder)
    ruleFinder = lambda x: x.split(" ")[3].split("\n")[0]
    labelsDf['Rule']=labelsDf['string'].apply(ruleFinder)
    
    #Compute a matrix of the labels in the right order
    #(This last step is really low because of the for loop, there's probably a faster way to do it, any idea ?)
    labelsMatrix = np.zeros([len(image_file_series),3])
    for i in range(0,len(image_file_series)):
        row = labelsDf[labelsDf['imagename']==image_file_series[i]]
        labelsMatrix[i,0] =float(row['Yau'])
        labelsMatrix[i,1] =float(row['Pitch'])
        labelsMatrix[i,2] =float(row['Rule'])
    
    
    return labelsMatrix
    
if __name__=='__main__':
	folder = '../Cropped/img_014b/GeneratedImgs/014b/Campagne_IR_Pose_20150710/cam1/1'

	image_size = 39  # Pixel width and height.
	pixel_depth = 255.0  # Number of levels per pixel.

	data = load_dataset(folder)
	labels = findLabelsMatrix(folder)
	print(labels)
