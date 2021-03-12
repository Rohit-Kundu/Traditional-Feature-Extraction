#GLCM or Gray Level Co-occurence Matrix extracts texture features from images

import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import cv2
from PIL import Image
import numpy as np
import csv
import os
import glob
import pandas as pd
from skimage.transform import resize
import time

tic=time.time()

PATCH_SIZE = 21

#GLCM will work on batch of images only if all the images are of same size. 
#Uncomment the following two lines of code and enter the dimensions of images you want if the dataset has inconsistent sizes of images:

#IMAGE_HEIGHT=
#IMAGE_WIDTH=

img_dir = ".../..../..." #Enter the directory where all the images are stored
data_path=os.path.join(img_dir,'*g')
files=glob.glob(data_path)

eo=len(files)

img = []
for f1 in files:
    data = cv2.imread(f1)
    img.append(data)

for i in range(eo):
        img[i] = cv2.cvtColor(img[i] , cv2.COLOR_BGR2GRAY)

        #Uncomment the following line for inconsistent size of images in dataset
        #img[i] = cv2.resize(img[i],(IMAGE_WIDTH,IMAGE_HEIGHT),interpolation=cv2.INTER_AREA)
        
        print("For image number:"+str(i+1)+'\n')
        image=img[i]
        print("Shape of image is: ",image.shape)
        
        # select some patches from grassy areas of the image
        grass_locations = [(1000,100), (980,100), (990,120), (985,150)]
        grass_patches = []
        for loc in grass_locations:
            grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,loc[1]:loc[1] + PATCH_SIZE])

        # select some patches from sky areas of the image
        sky_locations = [(417, 415), (427, 413), (420, 410), (422, 412)]
        sky_patches = []
        for loc in sky_locations:
            sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

        # compute some GLCM properties each patch
        xs = []
        ys = []
        bs = []
        cs = []
        ds = []

        for patch in (grass_patches + sky_patches):
            glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,symmetric=True, normed=True)
            xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
            ys.append(greycoprops(glcm, 'correlation')[0, 0])
            bs.append(greycoprops(glcm, 'contrast')[0, 0])
            cs.append(greycoprops(glcm, 'energy')[0, 0])
            ds.append(greycoprops(glcm, 'homogeneity')[0, 0])

        temp_xs=xs
        temp_ys=ys
        temp_bs=bs
        temp_cs=cs
        temp_ds=ds

        temp_xs.sort()
        temp_ys.sort()
        temp_bs.sort()
        temp_cs.sort()
        temp_ds.sort()

        xs_max=temp_xs[-1]
        ys_max=temp_ys[-1]
        bs_max=temp_bs[-1]
        cs_max=temp_cs[-1]
        ds_max=temp_ds[-1]

        xs_mean=sum(temp_xs)/len(xs)
        ys_mean=sum(temp_ys)/len(ys)
        bs_mean=sum(temp_bs)/len(bs)
        cs_mean=sum(temp_cs)/len(cs)
        ds_mean=sum(temp_ds)/len(ds)

        xs_var=np.var(np.array(temp_xs))
        ys_var=np.var(np.array(temp_ys))
        bs_var=np.var(np.array(temp_bs))
        cs_var=np.var(np.array(temp_cs))
        ds_var=np.var(np.array(temp_ds))

        df = pd.DataFrame()

        df['dissimilarity_max'] = [xs_max]
        df['dissimilarity_mean'] = xs_mean
        df['dissimilarity_var'] = xs_var

        df['correlation_max'] = ys_max
        df['correlation_mean'] = ys_mean
        df['correlation_var'] = ys_var

        df['contrast_max'] = bs_max
        df['contrast_mean'] = bs_mean
        df['contrast_var'] = bs_var

        df['energy_max'] = cs_max
        df['energy_mean'] = cs_mean
        df['energy_var'] = cs_var

        df['homogeneity_max'] = ds_max
        df['homogeneity_mean'] = ds_mean
        df['homogeneity_var'] = ds_var

        print(df)
        
        with open('GLCM_BreCaHAD_temp.csv','a+',newline='') as file:
            writer=csv.writer(file)
            
            if i==0:
                writer.writerow(['dissimilarity_max','dissimilarity_mean','dissimilarity_var','contrast_max','contrast_mean','contrast_var','energy_max','energy_mean',
                                 'energy_var','correlation_max','correlation_mean','correlation_var','homogeneity_max','homogeneity_mean','homogeneity_var'])
            
            writer.writerow([xs_max,xs_mean,xs_var,ys_max,ys_mean,ys_var,bs_max,bs_mean,bs_var,cs_max,cs_mean,cs_var,ds_max,ds_mean,ds_var])

#Uncomment the following section for features visualization
        '''
        # create the figure
        fig = plt.figure(figsize=(8, 8))

        # display original image with locations of patches
        ax = fig.add_subplot(3, 2, 1)
        ax.imshow(image, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        for (y, x) in grass_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
        for (y, x) in sky_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
        ax.set_xlabel('Original Image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('image')

        # for each patch, plot (dissimilarity, correlation)
        ax = fig.add_subplot(3, 2, 2)
        ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
                label='Grass')
        ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
                label='Sky')
        ax.set_xlabel('GLCM Dissimilarity')
        ax.set_ylabel('GLCM Correlation')
        ax.legend()

        # display the image patches
        for i, patch in enumerate(grass_patches):
            ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
            ax.imshow(patch, cmap=plt.cm.gray,
                      vmin=0, vmax=255)
            ax.set_xlabel('Grass %d' % (i + 1))

        for i, patch in enumerate(sky_patches):
            ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
            ax.imshow(patch, cmap=plt.cm.gray,
                      vmin=0, vmax=255)
            ax.set_xlabel('Sky %d' % (i + 1))


        # display the patches and plot
        fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
        plt.tight_layout()
        plt.show()
        '''
        
toc=time.time()
print('Computation time is : {} seconds'.format(str(toc-tic)))
