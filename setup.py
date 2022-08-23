# -*- coding: utf-8 -*-
"""
usage: setup.py [-n ROOTDIRNAME] [-h] [-c COLOR]
"""
import os
import shutil
import argparse
from distutils.dir_util import copy_tree
#%%
def clean_folder(path):
    shutil.rmtree(path)
    os.mkdir(path)
#%%
def setup_folders(root = "my_project", 
                  color_mode = "rgb"):
    """
    Creates necessary folders to store train data, model checkpoints, and other temporary folders
    """
    root = root.replace("-", "_")
    # define initial parameters
    initial_params = {"color_mode": color_mode,
                      "root": root,
                      "main_dir": os.getcwd(),
                      "version": 1,
                      "final_model_name": f"{root}_{color_mode}.onnx",
                      "batch_size": 6,
                      "num_epochs": 20,
                      "num_classes": 2,
                      "lr": 0.001}
    
    i = 1
    while True:
        try:
            os.mkdir(rf".\projects\{root}")
            break
        except FileExistsError:
            root = root.split("-")[0]
            root = root + f"-{i}"
            i = i + 1
    # Create dir
    #os.mkdir(rf".\projects\{root}")
    
    initial_params["root"] = root
    
    # Save parameters in a txt file
    with open(f'.\projects\{root}\parameters.txt', 'w') as f:
        for key, value in initial_params.items():
            f.write(f"{key}={value}\n")
    
    # define necessary folders
    necessary_folders = ["Labels", "Plots", "Labels\Images_train", "Labels\XML_train", "models", "models\checkpoints", "Results", "Test_Images", "model_scripts", "Data_Folder"]
    
    # create necessary folders
    for i in necessary_folders:
        if not os.path.isdir(f".\projects\{root}\{i}"):
            os.mkdir(rf".\projects\{root}\{i}")
    
    # transport model script
    shutil.copyfile(".\model_scripts\model.py", f".\projects\{root}\model.py") # change this later if more models are added
    copy_tree(".\scripts", f".\projects\{root}\model_scripts") # change this later if more models are added
    
    print("Done!")
    

def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Setup folders for pytorch object detection",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-n', '--rootDirName',
        help='Specifies the name of the root folder under the projects directory. By default is "my_project"',
        type=str,
        default="my_project"
    )
    
    parser.add_argument(
        '-c', '--color',
        help='Create an extra folder to store grey images or not. Either RGB or GREYSCALE. Defaults to RGB',
        type=str,
        default="rgb",
        choices={"rgb", "greyscale"}
    )
    
    args = parser.parse_args()
    
    # Begin setup
    setup_folders(args.rootDirName, args.color)
    
    
    
    
if __name__ == '__main__':
    main()