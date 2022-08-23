# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:00:27 2021

@author: metasystems
"""
#%%
import os
import sys
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#%%
import onnxruntime
import torch.onnx as onnx
import onnx
from PIL import Image
import numpy as np
import cv2
from torchvision import datasets, transforms, models #, references
import torch
#from utils.general import non_max_suppression
import torchvision
import matplotlib.pyplot as plt
#%%
print(onnxruntime.__version__)
onnxruntime.set_default_logger_severity(3) # Suppress logging of onnxruntime
#%%
def run_onnx(test_image_address, save_dir, model_path):
    # model path is where model is located
    # test image address is the location of the image
    # save_dir is where to save images with detected objects
    
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    #print(onnx.helper.printable_graph(onnx_model.graph))
    output = onnx_model.graph.output
    #print(output)
    #%%
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    #print(onnx.helper.printable_graph(onnx_model.graph))
    output = onnx_model.graph.output
    #print(output)
    #%%
    
    img = cv2.imread(test_image_address,1)
    #img = np.zeros((1496,2048,3)).astype(np.float32)
    #img = img.resize((1024,1024))
    #img = img.convert('RGB')
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    img.unsqueeze_(0)
    #%%
    ort_session = onnxruntime.InferenceSession(model_path)
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    #print(ort_outs)
    #%%
    img = torch.squeeze(img)
    img = img.permute(1,2,0)  # C,H,W_H,W,C, for drawing
    img = (img * 255).byte().data.cpu()  # * 255, float to 0-255
    img = np.array(img)  # tensor → ndarray
    im = img.astype(np.uint8).copy()
    for i in range(ort_outs[0].shape[0]):
        if ort_outs[2][i]>0.8:
            xmin = round(ort_outs[0][i][0].item())
            ymin = round(ort_outs[0][i][1].item())
            xmax = round(ort_outs[0][i][2].item())
            ymax = round(ort_outs[0][i][3].item())
       
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)            
    #filename = r"E:\WBC_prescan\Results\obj_onnx.png"
    #im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    #plt.imshow(im)
    #plt.show()
    #im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    file_name = test_image_address.split("\\")[-1]
    cv2.imwrite(rf"{save_dir}\{file_name}",im)
    #cv2.imshow('',im)
    #cv2.waitKey(0)

#%%
img_dir = r"D:\Shanxi_WBC_smear\Labels\Images"
save_dir = r"D:\shanxi_wbc_smear\Results"
model_path = r"D:\shanxi_wbc_smear\models\Shanxi_WBC_40X_v3.onnx" 
for dirpath, dirnames, filenames in os.walk(img_dir):
    for file in tqdm(filenames):
        run_onnx(os.path.join(dirpath,file), save_dir, model_path)