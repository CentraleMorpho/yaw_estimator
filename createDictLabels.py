import os
import random
import math
import pickle


def createDictFromLabelsFile():
	posesFile = '../Cropped/024_poseonly_normalised180.txt'
	dico = {}
	
	with open(posesFile,'r') as f:
			i=0
			for line in f:
				path = line.split(' ')[0]
				if(path.split('/')[0]=='020'):
					path = os.path.join("../Cropped/GeneratedImgs",path)
					#path = ' '.join((line.split(' ')[1],line.split(' ')[2],line.split(' ')[3][0:len(line.split(' ')[3])-2],path))
					dico[path]=[float(line.split(' ')[1]),float(line.split(' ')[2]),float(line.split(' ')[3])]
				elif(path.split('/')[0]=='022'):
					path = os.path.join("../Cropped/GeneratedImgs",path)
					#path = ' '.join((line.split(' ')[1],line.split(' ')[2],line.split(' ')[3][0:len(line.split(' ')[3])-2],path))
					dico[path]=[float(line.split(' ')[1]),float(line.split(' ')[2]),float(line.split(' ')[3][0:len(line.split(' ')[3])-2])]
				elif(path.split('/')[0]=='014b'):
					path = os.path.join("../Cropped/img_014b/GeneratedImgs",path)
					#path = ' '.join((line.split(' ')[1],line.split(' ')[2],line.split(' ')[3][0:len(line.split(' ')[3])-2], path))
					dico[path]=[float(line.split(' ')[1]),float(line.split(' ')[2]),float(line.split(' ')[3][0:len(line.split(' ')[3])-2])]
				print(i)
				i=i+1
				
	return dico
	
	


if __name__=='__main__':
	print('Making the dictionary...')
	dico = createDictFromLabelsFile()
	
    	with open('dictLabels' '.pkl', 'wb') as f:
        	pickle.dump(dico, f, pickle.HIGHEST_PROTOCOL)

