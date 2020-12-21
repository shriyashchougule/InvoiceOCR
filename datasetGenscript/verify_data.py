import h5py
import numpy as np
import os
import cv2
import random

with h5py.File('TrainImages.h5', 'r') as hfImages:
    with h5py.File('TrainMasks.h5', 'r') as hfMasks:
        print(len(hfImages.keys()))
        while 1:
            i = random.randint(0, len(hfImages.keys()))
            img = np.array(hfImages["{}".format(i)])
            msk = np.array(hfMasks[str(i)])
            cv2.imshow("img", img)
            cv2.imshow("msk", msk*60)
            print(i)
            cv2.waitKey(0)
'''
with h5py.File('ValidationImages.h5', 'r') as hfImages:
    with h5py.File('ValidationMasks.h5', 'r') as hfMasks:
        print(len(hfImages.keys()))
        while 1:
            i = random.randint(0, len(hfImages.keys()))
            img = np.array(hfImages["{}".format(i)])
            msk = np.array(hfMasks[str(i)])
            cv2.imshow("img", img)
            cv2.imshow("msk", msk*60)
            print(i)
            cv2.waitKey(0)
'''
