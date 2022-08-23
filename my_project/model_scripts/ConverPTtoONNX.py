# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:48:37 2022

@author: metasystems
"""
import torch
#%%
def convert_pt_to_onnx(model_name, onnx_name, 
                    color_mode="rgb", # color mode
                    img_size = (896, 1152)): # input image size
    """
    Converts pt model to onnx
    """

    # Convert PT Model into ONNX Model
    #model_name = "model_BM_grey_ver_2.pt" # torch PT model
    #model_path = f'D:\pytorch_bbox\projects\shanxi_bm_smear\models\{model_name}' # model folder location
    #onnx_name = "Shanxi_BM_40X_v2.onnx" # target ONNX model name
    model_path = f".\models\{model_name}"
    #%%
    if color_mode == "rgb":
        channels = 3
    elif color_mode == "greyscale":
        channels = 1
    else:
        print(f"input rgb or greyscale. Given value was: {color_mode}")
    
    model = torch.load(model_path) # load model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model.to("cpu") # for some reason does not work with cuda. Look into the issue later [BUG]
    
    # Create Dummy Input
    dummy_input = torch.randn(channels,img_size[0],img_size[1])
    dummy_input = dummy_input.unsqueeze_(0)
    dummy_input = dummy_input.to("cpu") # should be the same as model
    #dummy_input = dummy_input.to("cuda:0")
    
    #%%
    torch.onnx.export(model, 
                      dummy_input, 
                      f".\models\{onnx_name}", 
                      export_params = True,
                      opset_version=11) # export model to ONNX
