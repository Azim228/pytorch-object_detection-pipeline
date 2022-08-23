import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(os.getcwd())
#%%
from model_scripts import transforms as T
from model_scripts.engine import train_one_epoch

#%%

import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision
from xml.dom.minidom import parse
import xml
from torchvision.references.detection.engine import evaluate
from torchvision.references.detection import utils
import sys
import shutil
#%%
def plot_graph(losses, iou_stats, version, save_dir):
   print(f"Saving to {save_dir}")
   plt.figure(figsize=(12,8))
   plt.plot(losses)
   plt.xlabel("Epochs")
   plt.ylabel("Value")
   plt.title(f"Loss Values_version_ver{version}")
   #plt.savefig(f"./plots/iteration_{version}/losses_ver{version}.png")
   plt.savefig(f"{save_dir}/losses_ver{version}.png")
   
   plt.figure(figsize=(12,8))
   plt.title(f"Precision Metrics_ver{version}")
   plt.xlabel("Epochs")
   plt.ylabel("Value")
   
   for i, (label, values) in enumerate(iou_stats):
       if "small" not in label and "large" not in label and i <= 5:
           plt.plot(values, label = label)
           plt.legend()
       elif "small" not in label and "large" not in label and i >= 6:
           if i == 6:
               plt.savefig(f"{save_dir}/precisions_ver{version}.png")
               plt.figure(figsize=(12,8))
               plt.title(f"Recall Metrics_ver{version}")
               plt.xlabel("Epochs")
               plt.ylabel("Value")
           plt.plot(values, label = label)
   plt.legend()
   plt.savefig(f"{save_dir}/recalls_ver{version}.png")

#%%
def record_summary(iou_stats):
    with open("./summary.txt", mode="a") as file:
        file.write("\n")
        for label, values in iou_stats:
            file.write(f"{label} ={values[-1]: .3f}\n")
            
#%%
def clean_folder(path):
    shutil.rmtree(path)
    os.mkdir(path)
#%%
def read_txt(path):
    params = {}
    # Note: path should be full, including the target txt file
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.split("=") # split by '=' character
            # Record parameters to dictionary 
            if line[1].endswith("\n"):
                line[1] = line[1][:-1] # remove \n character at the end
            if line[0].endswith(" "):
                line[0] = line[0][:-1] # remove space at the end
            try:
                int(line[1])
                params[line[0]] = int(line[1])
            except ValueError:
                try:
                    float(line[1])
                    params[line[0]] = float(line[1])
                except ValueError:
                    params[line[0]] = line[1]
                
    return params
#%%
def write_to_txt(path, params):
    with open(path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}={value}\n")
#%%
losses = []
stats= [("Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", []),
        ("Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]", []),
        ("Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]", []),
        ("Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]", []),
        ("Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", []),
        ("Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]", []),
        ("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]", []),
        ("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]", []),
        ("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", []),
        ("Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]", []),
        ("Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", []),
        ("Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]", [])]
#%%
params = read_txt("parameters.txt")
version = params["version"]
color_mode = params["color_mode"]
final_model_name = params["final_model_name"]
batch_size = params["batch_size"]
num_epochs = params["num_epochs"]
num_classes = params["num_classes"]
main_dir = params["main_dir"]
root = params["root"]
lr = params["lr"]

os.chdir(f"{main_dir}\projects\{root}") # Set to current working directory

if os.path.isfile(f'.\models\\model_{root}_ver_{version}.pth'):
    print("You are rewriting existing model version! Starting a new iteration instead")
    version = version + 1
    params["version"] = version
    write_to_txt("parameters.txt", params)
#%%
print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(f"\nSTARTING TRAINING ITERATION {version}")
#%%
# load images and bbox    
imgs = list(sorted(os.listdir(os.path.join(r"./Labels", "Images_train"))))
bbox_xml = list(sorted(os.listdir(os.path.join(r"./Labels", "XML_train"))))
#%% Remove non-xml files from folder
temp = bbox_xml.copy()
for file in bbox_xml:
    if not file.lower().endswith("xml"):
        os.remove(f"./Labels/XML_train/{file}")
        temp.remove(file)
#%%
class MarkDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms = None, color_mode = 1):
        super().__init__()
        self.root = root
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images_train"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "XML_train"))))
        self.transforms = transforms
        
        if color_mode == "rgb":
            self.color_mode = 1
        else:
            self.color_mode = 0
 
    def __getitem__(self, idx):
        # load images and bbox        
        img_path = os.path.join(self.root, "Images_train", self.imgs[idx])
        bbox_xml_path = os.path.join(self.root, "XML_train", self.bbox_xml[idx])

        img = cv2.imread(img_path, self.color_mode)
        dom = parse(bbox_xml_path)
        data = dom.documentElement
        objects = data.getElementsByTagName('object')    
        if len(objects)>0:
            target = {}
            boxes = []
            labels = []
            for object_ in objects:
                
                
                bndbox = object_.getElementsByTagName('bndbox')[0]
                
                name = np.int64(object_.getElementsByTagName('name')[0].childNodes[0].nodeValue)
                labels.append(name)
                
                xmin = np.float64(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
                ymin = np.float64(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
                xmax = np.float64(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
                ymax = np.float64(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
                boxes.append([xmin, ymin, xmax, ymax])  
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)  
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["area"] = area
            iscrowd = torch.zeros((len(objects),), dtype=torch.int64)
    
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["iscrowd"] = iscrowd
    
     
            if self.transforms is not None:
                img,target = self.transforms(img,target)
     
            return img, target
        else:
            print(img_path)
             
    def __len__(self):
        return len(self.imgs)
#%%


def get_transform(train):
    transformations = []
    transformations.append(T.ToTensor())
    if train:
        transformations.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transformations)
#%%
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
#%%
# use our dataset and defined transformations
dataset = MarkDataset(".\Labels", get_transform(train = True), color_mode=color_mode)
dataset_test = MarkDataset(".\Labels",get_transform(train = False), color_mode=color_mode)

# split the dataset in train and test set
indices = torch.arange(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:int((len(indices)*0.9))])
dataset_test = torch.utils.data.Subset(dataset_test, indices[int((len(indices)*0.9)):])

# define training and validation data loaders
# Num_when training models in jupyter notebook The workers parameter can only be 0, otherwise an error will occur, which is commented out here
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False,
    collate_fn=utils.collate_fn)


# get the model using our helper function
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)  # Or get_object_detection_model(num_classes)
#%%

if version > 1:
    model.load_state_dict(torch.load(f'.\models\\model_{root}_ver_{version-1}.pth')) # load previous version
    
model.to(device)

# construct an optimizer
parameters = [p for p in model.parameters() if p.requires_grad]

# SGD
optimizer = torch.optim.SGD(parameters, lr=lr,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler, cos learning rate
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

#%%
def get_coco_stats(evaluator):
    for iou_type, coco_eval in evaluator.coco_eval.items():
        return coco_eval.stats
#%%
model.train()

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    print("train_one_epoch")
    metrics, loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    
    # record loss
    losses.append(loss) 
    
    # update the learning rate
    lr_scheduler.step() 
    
    evaluator = evaluate(model, data_loader_test, device=device)   
    
    # get precision and recall stats
    vals = get_coco_stats(evaluator)
    
    # append corresponding value to list of precisions and recalls
    for i in range(len(stats)):
        stats[i][1].append(vals[i]) 
    
    save_dir = f"./plots/iteration_{version}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    # plot and save figures
    plot_graph(losses, stats, version, save_dir)
    
    # record summary
    record_summary(stats)
        
    # Checkpoint every 10 epochs (to continue training before potential overfit)
    if epoch % 10 == 0:
        torch.save(model, f'.\models\\checkpoints\checkpoint_{epoch}.pt')
        torch.save(model.state_dict(), f'.\models\\checkpoints\checkpoint_{epoch}.pth')
    
    # Save latest version
    torch.save(model, f'.\models\\model_{root}_ver_{version}.pt')
    torch.save(model.state_dict(), f'.\models\\model_{root}_ver_{version}.pth')
    
    print('')
    print('==================================================')
    print('')
sys.exit()

























