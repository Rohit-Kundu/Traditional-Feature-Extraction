#GLRLM or Gray Level Run Length Matrix
#33 different features are extracted from this

import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np
from itertools import groupby

data = 0 
def read_img(path=" "):
        
        try:
            img = Image.open(path) 
            img = img.convert('L')
            self.data=np.array(img)
            
        except:
            img = None
            
def getGrayLevelRumatrix(array, theta):

            
            #array: the numpy array of the image
            #theta: Input, the angle used when calculating the gray scale run matrix, list type, can contain fields:['deg0', 'deg45', 'deg90', 'deg135']
            #glrlm: output,the glrlm result

            P = array
            x, y = P.shape
            min_pixels = np.min(P)   # the min pixel
            run_length = max(x, y)   # Maximum parade length in pixels
            num_level = np.max(P) - np.min(P) + 1   # Image gray level
    
            deg0 = [val.tolist() for sublist in np.vsplit(P, x) for val in sublist]   # 0deg
            deg90 = [val.tolist() for sublist in np.split(np.transpose(P), y) for val in sublist]   # 90deg
            diags = [P[::-1, :].diagonal(i) for i in range(-P.shape[0]+1, P.shape[1])]   #45deg
            deg45 = [n.tolist() for n in diags]
            Pt = np.rot90(P, 3)   # 135deg
            diags = [Pt[::-1, :].diagonal(i) for i in range(-Pt.shape[0]+1, Pt.shape[1])]
            deg135 = [n.tolist() for n in diags]
    
            def length(l):
                if hasattr(l, '_len_'):
                    return np.size(l)
                else:
                    i = 0
                    for _ in l:
                        i += 1
                    return i
    
            glrlm = np.zeros((num_level, run_length, len(theta)))   
            for angle in theta:
                for splitvec in range(0, len(eval(angle))):
                    flattened = eval(angle)[splitvec]
                    answer = []
                    for key, iter in groupby(flattened):  
                        answer.append((key, length(iter)))   
                    for ansIndex in range(0, len(answer)):
                        glrlm[int(answer[ansIndex][0]-min_pixels), int(answer[ansIndex][1]-1), theta.index(angle)] += 1   
            return glrlm

 # The gray scale run matrix is only the measurement and statistics of the image pixel information. In the actual use process, the generated
  #   The gray scale run matrix is calculated to obtain image feature information based on the gray level co-occurrence matrix.
  #   First write a few common functions to complete the calculation of subscripts i and j (calcuteIJ ()), multiply and divide according to the specified dimension (apply_over_degree ())
   #  And calculate the sum of all pixels (calcuteS ())
        
            
def apply_over_degree(function, x1, x2):
        rows, cols, nums = x1.shape
        result = np.ndarray((rows, cols, nums))
        for i in range(nums):
                #print(x1[:, :, i])
                result[:, :, i] = function(x1[:, :, i], x2)
               # print(result[:, :, i])
                result[result == np.inf] = 0
                result[np.isnan(result)] = 0
        return result 
def calcuteIJ (rlmatrix):
        gray_level, run_length, _ = rlmatrix.shape
        I, J = np.ogrid[0:gray_level, 0:run_length]
        return I, J+1

def calcuteS(rlmatrix):
        return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]
    
    #The following code realizes the extraction of 11 gray runoff matrix features
   
    #1.SRE
def getShortRunEmphasis(rlmatrix):
            I, J = calcuteIJ(rlmatrix)
            numerator = np.apply_over_axes(np.sum, apply_over_degree(np.divide, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
            S = calcuteS(rlmatrix)
            return numerator / S
    #2.LRE
def getLongRunEmphasis(rlmatrix):
        I, J = calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, apply_over_degree(np.multiply, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
        S = calcuteS(rlmatrix)
        return numerator / S
    #3.GLN
def getGrayLevelNonUniformity(rlmatrix):
        G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
        numerator = np.apply_over_axes(np.sum, (G*G), axes=(0, 1))[0, 0]
        S = calcuteS(rlmatrix)
        return numerator / S
    # 4. RLN
def getRunLengthNonUniformity(rlmatrix):
            R = np.apply_over_axes(np.sum, rlmatrix, axes=0)
            numerator = np.apply_over_axes(np.sum, (R*R), axes=(0, 1))[0, 0]
            S = calcuteS(rlmatrix)
            return numerator / S

        # 5. RP
def getRunPercentage(rlmatrix):
            gray_level, run_length,_ = rlmatrix.shape
            num_voxels = gray_level * run_length
            return calcuteS(rlmatrix) / num_voxels

        # 6. LGLRE
def getLowGrayLevelRunEmphasis(rlmatrix):
            I, J = calcuteIJ(rlmatrix)
            numerator = np.apply_over_axes(np.sum, apply_over_degree(np.divide, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
            S = calcuteS(rlmatrix)
            return numerator / S

        # 7. HGL   
def getHighGrayLevelRunEmphais(rlmatrix):
        I, J = calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, apply_over_degree(np.multiply, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
        S = calcuteS(rlmatrix)
        return numerator / S

        # 8. SRLGLE
def getShortRunLowGrayLevelEmphasis(rlmatrix):
        I, J = calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, apply_over_degree(np.divide, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = calcuteS(rlmatrix)
        return numerator / S
    # 9. SRHGLE
def getShortRunHighGrayLevelEmphasis(rlmatrix):
        I, J = calcuteIJ(rlmatrix)
        temp = apply_over_degree(np.multiply, rlmatrix, (I*I))
        print('-----------------------')
        numerator = np.apply_over_axes(np.sum, apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = calcuteS(rlmatrix)
        return numerator / S
 
    # 10. LRLGLE
def getLongRunLow(rlmatrix):
        I, J = calcuteIJ(rlmatrix)
        temp = apply_over_degree(np.multiply, rlmatrix, (J*J))
        numerator = np.apply_over_axes(np.sum, apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = calcuteS(rlmatrix)
        return numerator / S
 
    # 11. LRHGLE
def getLongRunHighGrayLevelEmphais(rlmatrix):
        I, J = calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum,apply_over_degree(np.multiply, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = calcuteS(rlmatrix)
        return numerator / S

#import getGrayRumatrix
from PIL import Image
import numpy as np
from itertools import groupby
import csv
import warnings
warnings.filterwarnings("ignore")
import cv2
import os
import glob
import time

tic=time.time()

img_dir=".../..../..." #Enter the directory where all the images are stored
data_path=os.path.join(img_dir,'*g')
files=glob.glob(data_path)
i=0
for path in files:
        data = 0
        img = Image.open(path) 
        img = img.convert('L')
        data=np.array(img)

        DEG = [['deg0'], ['deg45'], ['deg90'], ['deg135']]

        with open('GLRLM.csv','a+',newline='',encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                
                if i==0:
                        csv_writer.writerow(['deg0_SRE','deg45_SRE','deg90_SRE','deg135_SRE','deg0_LRE','deg45_LRE','deg90_LRE','deg135_LRE',
                                             'deg0_GLN','deg45_GLN','deg90_GLN','deg135_GLN','deg0_RLN','deg45_RLN','deg90_RLN','deg135_RLN',
                                             'deg0_RP','deg45_RP','deg90_RP','deg135_RP','deg0_LGLRE','deg45_LGLRE','deg90_LGLRE','deg135_LGLRE',
                                             'deg0_HGL','deg45_HGL','deg90_HGL','deg135_HGL','deg0_SRLGLE','deg45_SRLGLE','deg90_SRLGLE','deg135_SRLGLE',
                                             'deg0_SRHGLE','deg45_SRHGLE','deg90_SRHGLE','deg135_SRHGLE','deg0_LRLGLE','deg45_LRLGLE','deg90_LRLGLE','deg135_LRLGLE',
                                             'deg0_LRHGLE','deg45_LRHGLE','deg90_LRHGLE','deg135_LRHGLE'])
                
                print("Processing Image",i+1)
                i+=1
                SRE_l=[]
                LRE_l=[]
                GLN_l=[]
                RLN_l=[]
                RP_l=[]
                LGLRE_l=[]
                HGL_l=[]
                SRLGLE_l=[]
                SRHGLE_l=[]
                LRLGLE_l=[]
                LRHGLE_l=[]
                for deg in DEG:
                    now_deg = deg[0]
                    test_data = getGrayLevelRumatrix(data,deg)
                    

                    #1
                    SRE = getShortRunEmphasis(test_data) 
                    SRE = np.squeeze(SRE)
                    SRE_l.append(SRE)
                 

                    #2
                    LRE = getLongRunEmphasis(test_data)
                    LRE = np.squeeze(LRE)
                    LRE_l.append(LRE)
                    
                    #3
                    GLN = getGrayLevelNonUniformity(test_data)
                    GLN = np.squeeze(GLN)
                    GLN_l.append(GLN)
                    
                    #4
                    RLN = getRunLengthNonUniformity(test_data)
                    RLN = np.squeeze(RLN)
                    RLN_l.append(RLN)
                    
                    #5
                    RP = getRunPercentage(test_data)
                    RP = np.squeeze(RP)
                    RP_l.append(RP)
                    
                    #6
                    LGLRE = getLowGrayLevelRunEmphasis(test_data)
                    LGLRE = np.squeeze(LGLRE)
                    LGLRE_l.append(LGLRE)
                    
                    #7
                    HGL = getHighGrayLevelRunEmphais(test_data)
                    HGL = np.squeeze(HGL)
                    HGL_l.append(HGL)
                    
                    #8
                    SRLGLE = getShortRunLowGrayLevelEmphasis(test_data)
                    SRLGLE = np.squeeze(SRLGLE)
                    SRLGLE_l.append(SRLGLE)
                    
                    #9
                    SRHGLE = getShortRunHighGrayLevelEmphasis(test_data)
                    SRHGLE = np.squeeze(SRHGLE)
                    SRHGLE_l.append(SRHGLE)
                    
                    #10
                    LRLGLE = getLongRunLow(test_data)
                    LRLGLE = np.squeeze(LRLGLE)
                    LRLGLE_l.append(LRLGLE)
                    
                    #11
                    LRHGLE = getLongRunHighGrayLevelEmphais(test_data)
                    LRHGLE = np.squeeze(LRHGLE)
                    LRHGLE_l.append(LRHGLE)
            

                csv_writer.writerow(SRE_l+LRE_l+GLN_l+RLN_l+RP_l+LGLRE_l+HGL_l+SRLGLE_l+SRHGLE_l+LRLGLE_l+LRHGLE_l)

toc=time.time()
print("Computation time is: {} minutes.".format(str((toc-tic)/60)))

