import numpy as np
import cv2 as cv
import os
import linecache
import scipy
import scipy.misc
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from time import sleep
from tqdm import tqdm
from numba import jit, int32
import time
import scipy
import pandas as pd
from tensorboardX import SummaryWriter


save_path = 'E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\save.txt'
image_path = "E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\JPEGImages"
image_mask_path = "E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\"
image_outputpath = "E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\output-file\\"

"""
Read the folder where the dataset to be processed is located, 
and store all file names in the save.txt file by line in order
"""

f = open(save_path, 'w')
for filename in os.listdir("E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\JPEGImages\\"):
    f.write(str(filename))
    f.write('\n')
f.close()

#Get the content of the specified line in the txt file, here in order to get the file name of the image that needs to be processed currently

def get_txt_line(file, nums_line):
    return linecache.getlines(file, nums_line).strip()

def check_mkdir(path):
    mask_folder = os.path.exists(path)
    if not mask_folder:
        os.makedirs(path)


#Use a 4*4 mask to remove low-pass frequency components
def remove_low_freq(img_num_total, mask):

    for img_num in tqdm(range(0,img_num_total)):#Read reads each image in sequence, img_num_total is the total number of images in the folder
        n = 0
        img_nu = img_num + n

        #Get the file name of the specified line in the storage path file, here is the image name in the next dataset to be processed
        f = open(save_path)
        for num, line in enumerate(f):#line is the name of the line
            if num == img_nu:
                break
        img = Image.open(
            image_path
            + '\\'
            + line.strip('\n')).convert(
            'L')


        #img.save(image_outputpath + 'output_' + line.strip('\n'))
        f = np.fft.fft2(img)  # Fast Fourier Transform
        fshift = np.fft.fftshift(f)  ## shift for centering 0.0 (x,y)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))

        rows = np.size(img, 0)  # taking the size of the image
        cols = np.size(img, 1)
        crow, ccol = rows // 2, cols // 2
        fshift[crow - (mask//2) :crow + (mask//2), ccol - (mask//2):ccol + (mask//2)] = 0

        f_ishift = np.fft.ifftshift(fshift)
        img_backTrans = np.fft.ifft2(f_ishift)  ## shift for centering 0.0 (x,y)
        img_backTrans = np.abs(img_backTrans)
        img_backTrans = Image.fromarray(img_backTrans)
        if img_backTrans.mode != 'RGB':
            img_backTrans = img_backTrans.convert('RGB')

            mask_folder_path = image_mask_path + 'jpeg_mask_%d'%mask
            #Check if a folder with this mask exists
            check_mkdir(mask_folder_path)

            img_backTrans.save(mask_folder_path+ '\\img_backTrans_'
            + line.strip('\n'))
    return img_num_total


#Read the image and do the subtract operation
def subtract_two(img_num_total, mask1, mask2):

    for img_num in tqdm(range(0, img_num_total)):
        n = 0
        img_nu = img_num + n
        # Get the file name of the specified line in the storage path file, here is the image name in the next dataset to be processed
        f = open(save_path)
        for num, line in enumerate(f):  # line is the name of the line
            if num == img_nu:
                break
        img_mask1 = cv.imread(image_mask_path
                              + 'jpeg_mask_%d'%mask1
                              + '\\img_backTrans_%s'%line.strip('\n'))
        img_mask2 = cv.imread(image_mask_path
                              + 'jpeg_mask_%d'% mask2
                              + '\\img_backTrans_%s'%line.strip('\n'))
        subtracted_of_2imgs = cv.subtract(img_mask1, img_mask2)
        sub_path = image_mask_path + 'sub_mask_%d'%mask1 + '_%d'%mask2
        check_mkdir(sub_path)
        cv.imwrite(sub_path + '\\img_sub_%s'%line.strip('\n'), subtracted_of_2imgs)
    return img_num_total

"""
Read all images, transform and subtract them
"""
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    img_num_total = 0
    # Read the folder of the pictures to be processed, and save all the picture names in a folder
    f = open(save_path, 'w')
    for filename in os.listdir(image_path):
        f.write(str(filename))
        f.write('\n')
        img_num_total+=1
    f.close()

    mask1 = 4
    mask2 = 100
    t0 = time.time()
    remove_low_freq(img_num_total, mask1)
    remove_low_freq(img_num_total, mask2)
    subtract_two(img_num_total, mask1, mask2)
    t1 = time.time()
    print("Time: ", (t1 - t0))

