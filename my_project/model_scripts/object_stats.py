# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:50:30 2022

@author: metasystems
"""

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#%%
from PIL import Image
import pandas as pd
import numpy as np
import cv2
#from utils.general import non_max_suppression
import matplotlib.pyplot as plt
import os
import shutil
import math
from tqdm import tqdm
#%%
def clean_folder(path):
    shutil.rmtree(path)
    os.mkdir(path)
#%%
def get_object_size(slide_name, input_dir, mode = "circular"):
    input_dir = rf"{input_dir}\{slide_name}"
    input_img_paths = sorted(
        [
         os.path.join(input_dir, fname)
         for fname in os.listdir(input_dir)
         if fname.endswith(".tif")
         ]
    )
    
    input_csv_paths = sorted(
        [
         os.path.join(input_dir, fname)
         for fname in os.listdir(input_dir)
         if fname.endswith(".csv")
         ]
    )
    boxes_tot = []
    for csv in input_csv_paths:
        filename_csv = os.path.basename(csv)
        vsbox = pd.read_csv(input_dir + '\\'+ filename_csv, sep=';', usecols = [*range(1,5)], names = ['cx','cy','xrad','yrad'], header = None)
        boxes =[]
        for row in vsbox.iterrows():
            row=row[1]
            row = row.tolist()
            if mode == "circular":
                x_min = int(row[0]-row[2])
                y_min = int(row[1]-row[3])
                x_max = int(row[0]+row[2])
                y_max = int(row[1]+row[3])
            elif mode == "rectangular":
                x_min = int(row[0])
                y_min = int(row[1])
                x_max = int(row[2])
                y_max = int(row[3])
            box = [x_min,y_min,x_max,y_max]
            #boxes.append(box)
        boxes_tot.append(box)
    obj_count = len(boxes_tot)
    
    
    
    stats = []
    for bbox in boxes_tot:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        #stats.append((width, height, width*height))
        stats.append([width, height, width*height, math.sqrt(width*height)])
    return np.array(stats)
#%%
def get_stats(input_dir, save_dir):
    """
    Get the distribution of height, width, area and bbox size of the dataset
    
    -input_dir:
        where are the fodlers with data (tiff and csv files)
    -save_dir:
        where to save plots
    """
    
    for dirpath, dirnames, filenames in os.walk(input_dir):
        print(dirnames)
        count = 0
        arr = get_object_size(dirnames[0], input_dir, "rectangular")
        for file in tqdm(dirnames[1:]):
            #print(arr)
            arr = np.concatenate((arr, get_object_size(file, input_dir, "rectangular")), axis = 0)
        print(f"\nTotal Object Count is: {len(arr)}\n")
        arr = np.array(arr)
        #print(arr[:,0])
        titles = {0:"Width", 1:"Height", 2:"Area", 3:"BBox Size"}
        
        
        for i in range(4):
            freq, bins = np.histogram(arr[:,i], bins = "auto")
            relative_width = (np.max(bins) - np.min(bins)) / len(bins)
            #plt.hist(arr[:,i], bins = "auto")
            plt.bar(bins[:-1], freq, align="edge", ec="k", color='red', width=relative_width)
            plt.title(f"Histogram on {titles[i]} with AUTO bins")
            plt.grid()
            plt.savefig(f"{save_dir}\hist_{titles[i]}.png")
            plt.show()
        print(f"Frequency: {freq}\nBins: {bins}\nTotal Cell Count: {len(arr)}")
        break
#%%
