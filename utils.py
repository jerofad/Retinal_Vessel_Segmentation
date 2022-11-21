""" Rename all the dfolde rin the UI dataset"""
# import os

# work_dir = '/Users/jeremiahfadugba/Projects/RetinaSegmentation/Retinal_Vessel_Segmentation/datasets/UCH'
# for i, filename in enumerate(os.listdir(work_dir)):
#     os.rename(f"{work_dir}/{filename}",f"{work_dir}/{str(i)}.jpg")
import cv2
import numpy as np
import torch
from models import *
from glob import glob
import matplotlib.pyplot as plt
from dataset import *

""" Visualize"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

H = 224
W = 224
size = (W, H)


def read_image(x,y):
    image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)

    mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    mask = cv2.resize(mask, size)

    return image, mask


def process_image(image, mask):
    x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
    x = x/255.0
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)

    """ Reading mask """
    
    y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
    y = y/255.0
    y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
    y = y.astype(np.float32)
    y = torch.from_numpy(y)
    y = y.to(device)

    return x, y


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask


# # FPN 
# state_fname = f"results/No_Augmentation_FPN-chase" #name of file 
# model_fpn = FPN.to(device)
# model_fpn.load_state_dict(torch.load(state_fname, map_location=device))

# # UNET
# state_fname = f"results/No_Augmentation_Unet-chase" #name of file 
# model_unet = Unet.to(device)
# model_unet.load_state_dict(torch.load(state_fname, map_location=device))

# #DeepLab

# state_fname = f"results/No_Augmentation_Unet-chase" #name of file 
# model_deeplab = DeepLab.to(device)
# model_deeplab.load_state_dict(torch.load(state_fname, map_location=device))

# Test each model on 5 datas amples.

import random

# samples = random.sample(zip(test_x, test_y), 10)

data_str = "chase"
test_x, test_y = load_chase_images()

def return_pred(x, model):
    with torch.no_grad():
        """ Prediction and Calculating FPS """
        pred_y = model.predict(x)

        pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
        pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
        pred_y = pred_y > 0.5
        pred_y = np.array(pred_y, dtype=np.uint8)

    return pred_y

models = [DeepLab, FPN, Unet]
model_names = ["DeepLab", "FPN", "Unet"]
# x_sub, y_sub = zip(*random.sample(list(zip(test_x, test_y)), 5))
data, label = zip(*random.sample(list(zip(test_x, test_y)), 5))
# print(len(x), len(y))

for i, (x,y) in enumerate(zip(data, label)) :

    """ Extract the name """
    name = x.split("/")[-1].split(".")[0]

    """ read and process image """
    image, mask = read_image(x, y)
    x, y = process_image(image, mask)

    """ get predictions """
    
    predictions = {}
    for  (modelname, model_constructor) in zip(model_names, models):
        try: 
            state_fname = f"results/No_Augmentation_{modelname}-{data_str}" #name of file 
            model = model_constructor().to(device)
            model.load_state_dict(torch.load(state_fname, map_location=device))
        except FileNotFoundError:
            print(f"{state_fname} not found, continuing...")
            continue
        prediction = return_pred(x, model)
        predictions[modelname] = prediction

    """ Visualization """
    fig, axes = plt.subplots(nrows=1, ncols=2+len(predictions), figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].title.set_text(f"Original Image")
    axes[0].axis("off"); 
    axes[1].imshow(mask_parse(mask))
    axes[1].title.set_text(f"Ground Truth")
    axes[1].axis("off")
    for i, (k, v) in enumerate(predictions.items()):
        print
        axes[i+2].imshow(mask_parse(predictions[k])*255, cmap="seismic")
        axes[i+2].title.set_text(f"{k} Prediction")
        axes[i+2].axis("off")
    
    # save image
    fig.savefig(f"results/{data_str}/{name}.png") 