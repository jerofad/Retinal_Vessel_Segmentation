import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
from models import *
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.metrics import auc, plot_precision_recall_curve, roc_auc_score



""" Hyperparameters """
H = 512
W = 512
size = (W, H)
data_path = '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/'
result_path = '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/results/'

model_names = ["DeepLab", "FPN", "Unet"]
dataset_str = ["chase", "fives"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {}


def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5  # threshold
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    score_roc_auc = roc_auc_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_roc_auc]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask


""" DATSET  """

def load_Fives_images():

    valid_x = sorted(glob(data_path+"FIVES/test/Original/*"))
    valid_y = sorted(glob(data_path+"FIVES/test/Ground truth/*"))

    return valid_x, valid_y


def load_chase_images():

    valid_x = sorted(glob(data_path+"CHASE_DB1/test_data/*"))
    valid_y = sorted(glob(data_path+"CHASE_DB1/test_label/*"))

    return valid_x, valid_y


# def load_ui_images():
#     valid_x = sorted(glob(config['data_path']+ "CHASE_DB1/test_data/*"))
  
#     return valid_x


def read_image(x,y):
    image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
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

models = [DeepLab, FPN, Unet]

for model_constructor, model_name in zip(models, model_names):

    for data_str in dataset_str:

        if not os.path.isdir(f"{result_path}{data_str}/{model_name}"):
            os.makedirs(f"{result_path}{data_str}/{model_name}")

        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        time_taken = []
        try:
            state_fname = f"{result_path}No_Augmentation_{model_name}-{data_str}" #name of file 
            model = model_constructor().to(device)
            model.load_state_dict(torch.load(state_fname, map_location=device))

        except FileNotFoundError:
            print(f"{state_fname} not found, continuing...")
            continue

        if data_str == "chase":
            test_x, test_y = load_chase_images()
        elif data_str == "fives":
            test_x, test_y = load_Fives_images()
        else:
            raise AttributeError(f"{data_str} not valid; choices are fives, chase")

        for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
            """ Extract the name """
            name = x.split("/")[-1].split(".")[0]

            """ read and process image """
            image, mask = read_image(x, y)
            x, y = process_image(image, mask)

            with torch.no_grad():
                """ Prediction and Calculating FPS """
                start_time = time.time()
                pred_y = model.predict(x)
                # pred_y = torch.sigmoid(pred_y)
                total_time = time.time() - start_time
                time_taken.append(total_time)

                score = calculate_metrics(y, pred_y)
                metrics_score = list(map(add, metrics_score, score))
                pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
                pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
                pred_y = pred_y > 0.5
                pred_y = np.array(pred_y, dtype=np.uint8)

            """ Saving masks """
            ori_mask = mask_parse(mask)
            pred_y = mask_parse(pred_y)
            line = np.ones((size[1], 10, 3)) * 128

            cat_images = np.concatenate(
                [image, line, ori_mask, line, pred_y * 255], axis=1
            )
            cv2.imwrite(f"{result_path}{data_str}/{model_name}/{name}.png", cat_images)


        jaccard = metrics_score[0]/len(test_x)
        f1 = metrics_score[1]/len(test_x)
        recall = metrics_score[2]/len(test_x)
        precision = metrics_score[3]/len(test_x)
        acc = metrics_score[4]/len(test_x)
        roc_auc = metrics_score[5]/len(test_x)

        print(f"Results from {model_name} for {data_str}\n\n" 
            f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f}\n"
            f"Precision: {precision:1.4f} - Acc: {acc:1.4f} - ROC-AUC: {roc_auc:1.4f}")

        fps = 1/np.mean(time_taken)
        print("FPS: ", fps)