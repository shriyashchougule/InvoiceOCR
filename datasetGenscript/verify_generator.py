import h5py
#import matplotlib.pyplot as plt
#import matplotlib.pylab as plb
import numpy as np
import os
import cv2
import random

from custom_batch_generator_hpf5 import myDataGenerator

imgDatasetPath = "./TrainImages.h5"
mskDatasetPath = "./TrainMasks.h5"
batch_size = 4
mDG = myDataGenerator(imgDatasetPath, mskDatasetPath, batch_size)
print("number of batchs: {}".format(len(mDG)))

#Images, Masks  = mDG[0]
#print(Images.shape, Masks.shape)

for i in range(0,len(mDG)):
    Images, Masks = mDG[i]

    batchImg = np.array( Images[0], dtype=np.uint8)
    batchMsk = np.array( Masks[0], dtype=np.uint8)
    for j in range(1, batch_size):
        batchImg = np.hstack(( batchImg, np.array(Images[j]*255, dtype=np.uint8) ))
        batchMsk = np.hstack(( batchMsk, np.array(Masks[j], dtype=np.uint8) ))
    cv2.imshow("imgbatch", batchImg)
    cv2.imshow("mskbatch", batchMsk*60)
    cv2.waitKey(80)
