# First project : DL model on labeled images

Code based on python-tensorflow version of vgg. Intended to predict the yaw-pitch-roll angles relative to the camera from face cropped images.

Please follow the instructions to make it work :
Create a folder 'Cropped' next to the 'yaw_estimator' repo, and extract all the tar files in it.

Then go to the yaw-estimator repo and run :

```
python mydataset.py
```

This will create txt files for training and validation sets, containing the relative paths of the images.

Finally, you run 

```
python createDictLabels.py
```
This creates the dictionary of keys the images paths and with values the labels
