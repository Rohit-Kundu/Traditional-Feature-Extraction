#13 texture features are extracted from this

import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
import csv
import time

tic = time.time()

# function to extract haralick textures from an image
def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean  = textures.mean(axis=0)
    return ht_mean

# load the training dataset
train_path  = ".../..../..." #Enter the directory where all the images are stored
train_names = os.listdir(train_path)


# empty list to hold feature vectors and train labels
train_features = []
train_labels   = []

# loop over the training dataset
print ("[STATUS] Started extracting haralick textures..")
cur_path = os.path.join(train_path, '*g')
cur_label = train_names
i = 0
with open('Haralick_BreaKHis_temp.csv','a+',newline='') as obj:
                writer = csv.writer(obj)
                if i==0:
                        writer.writerow(['Haralick1','Haralick2','Haralick3','Haralick4','Haralick5','Haralick6','Haralick7','Haralick8','Haralick9',
                                         'Haralick10','Haralick11','Haralick12','Haralick13'])
                for file in glob.glob(cur_path):
                    print ("Processing Image - {} in {}".format(i, cur_label[i]))
                    #read the training image
                    image=cv2.imread(file)

                    #convert the image to grayscale
                    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    
                    #extract haralick texture from image
                    features=extract_features(gray)
                    #print(features)
                    
                    #append the feature vector and label
                    train_features.append(features)
                    train_labels.append(cur_label[i])

                    
                    writer.writerow(features)

                    #show loop update
                    i+=1

    
# have a look at the size of our feature vector and labels
print ("Training features: {}".format(np.array(train_features).shape))
print ("Training labels: {}".format(np.array(train_labels).shape))

toc = time.time()
print("Computation time is {} minutes.".format((toc-tic)/60))
