# Pytorch Object Detection Pipeline for Easy Model Training

This repository focuses on delivering easy start for object detection using state-of-the-art object detection solutions.

## Create a new Project
```bash
python setup.py [-n project_name] [-c --color {rgb, greyscale}]
```
Setup.py script creates a new project and specifies its name, color mode and transports all necessary scripts for the model to work.

## Prepare Dataset for Training
The data should be in Image and XML formats. Images can be in JPG, PNG or TIF formats. It is compulsory to follow the format of markdown used by LabelImg application. Read more about LabelImg: https://github.com/heartexlabs/labelImg

This repository has a simple example of a trained model and the folder structure.

Maake sure that data is clean and free of corruption, as the data cleaning is not yet part of this repository.

## Run the model
Go to the project root folder and run from terminal:
```bash
python model.py
```

To edit the model parameters go parameters.txt inside the project folder. It has the following parameters:
- color_mode = rgb
- root = project_name
- main_dir = D:\github_project
- version = 1
- final_model_name = project_name_rgb.onnx
- batch_size = 6
- num_epochs = 20
- num_classes = 2
- lr = 0.001

Tweaking and changing parameters can be done through this txt file or directly in model.py script. With future additions, the number of supported parameters would be increased.

## Project Folder structure
- Data_Folder: Folder for raw data. Mainly used as a backup to prevent data loss
- Labels: Note that script automatically splits the dataset into train/validation
  - Images_train: Training Images
  - XML_train: XML Annotations
- model_scripts: Necessary scripts. Do not edit
- models: Trained models are stored here
- Plots: This folder contains loss value history for each version and saves the histogram of bounding box sizes
- Results & Test_images: Reserved for future functionality
- model.py: Runs the model
- parameters.txt: Contains model hyperparameters
- summary.txt: Records model training results for each epoch

## Project Status - TODO
- Add more model backbones aside from resnet50 and RCNN Object Detection
- Add support for Model Checkpoints
- Add better model progress bars and graphs
- Add checking model inference utility
