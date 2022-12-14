""" Dataset Class for Fundus Image Segmentation. Dataset Implememnted includes:
    FIVES Dataset:
    DRIVE Dataset:
    STARE Dataset:
    CHASEDB Dataset:
"""

# Import statements
import os
import numpy as np
import cv2
import PIL.Image
from PIL.Image import open
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
from torchvision.transforms import functional as tf
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from augmentaions import *
from glob import glob

from utils import clahe_equalized
# pipeline Transform


def pipeline_tranforms():
    return Compose([RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    Fix_RandomRotation(),
                    ])


class Fix_RandomRotation(object):

    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        return tf.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class FundusDataset(Dataset):
    """ This wworks for DRIVE and FIVES"""

    def __init__(self, images_path, masks_path, config, pre_process=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transforms = pipeline_tranforms()
        self.process = pre_process
        self.config = config
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        if self.process:
            image = clahe_equalized(image)
        image = cv2.resize(image, self.config['size'])
        image = image / 255.0  # (512, 512, 3) Normalizing to range (0,1)
        image = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,  self.config['size'])
        mask = mask / 255.0  # (512, 512)
        mask = np.expand_dims(mask, axis=0)  # (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        # common transform
        seed = torch.seed()
        torch.manual_seed(seed)
        image = self.transforms(image)
        torch.manual_seed(seed)
        mask = self.transforms(mask)

        return image, mask

    def __len__(self):
        return self.n_samples


def load_Fives_images(config):
    # remove the last db file
    train_x = sorted(glob(config['data_path'] + "FIVES/train/Original/*"))[:-1]
    train_y = sorted(glob(config['data_path'] + "FIVES/train/Ground truth/*"))

    valid_x = sorted(glob(config['data_path'] + "FIVES/test/Original/*"))
    valid_y = sorted(glob(config['data_path'] + "FIVES/test/Ground truth/*"))

    return train_x, train_y, valid_x, valid_y


def load_chase_images(config):
    train_x = sorted(glob(config['data_path'] + "CHASE_DB1/train_data/*"))
    train_y = sorted(glob(config['data_path'] + "CHASE_DB1/train_label/*"))

    valid_x = sorted(glob(config['data_path'] + "CHASE_DB1/test_data/*"))
    valid_y = sorted(glob(config['data_path'] + "CHASE_DB1/test_label/*"))

    return train_x, train_y, valid_x, valid_y


def load_drive_images(config):
    train_x = sorted(glob(config['data_path'] + "DRIVE/train_data/*"))
    train_y = sorted(glob(config['data_path'] + "DRIVE/train_label/*"))

    valid_x = sorted(glob(config['data_path'] + "DRIVE/test_data/*"))
    valid_y = sorted(glob(config['data_path'] + "DRIVE/test_label/*"))

    return train_x, train_y, valid_x, valid_y


def load_stare_images(config):
    train_x = sorted(glob(config['data_path'] + "STARE/train_data/*"))
    train_y = sorted(glob(config['data_path'] + "STARE/train_label/*"))

    valid_x = sorted(glob(config['data_path'] + "STARE/test_data/*"))
    valid_y = sorted(glob(config['data_path'] + "STARE/test_label/*"))

    return train_x, train_y, valid_x, valid_y


def get_data(config):

    if config['data'] == "fives":
        train_x, train_y, valid_x, valid_y = load_Fives_images(config)
    elif config['data'] == "chase":
        config['batch_size'] = 2
        train_x, train_y, valid_x, valid_y = load_chase_images(config)
    elif config['data'] == "drive":
        config['batch_size'] = 2
        train_x, train_y, valid_x, valid_y = load_drive_images(config)
    elif config['data'] == "stare":
        config['batch_size'] = 2
        train_x, train_y, valid_x, valid_y = load_stare_images(config)
    else:
        raise AttributeError(
            f"{config['data']} not valid; choices are fives, chase, drive, stare")

    return train_x, train_y, valid_x, valid_y


def get_loader(config):

    train_x, train_y, _, _ = get_data(config)

    if config['aug_str'] == 'No_Augmentation':  # No preprocessing.
        full_dataset = FundusDataset(train_x, train_y, config)
    else:  # pre-process image
        full_dataset = FundusDataset(
            train_x, train_y, config, pre_process=True)

    # Split dataset into train and validation
    validation_split = .2
    shuffle_dataset = True
    random_seed = config['random_seed']
    # Creating data indices for training and validation splits:
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # train_dataset = FundusDataset(train_x, train_y, config)
    # val_dataset = FundusDataset(valid_x, valid_y, config)

    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=config['batch_size'], pin_memory=True,
                                               sampler=train_sampler, drop_last=True, num_workers=config['num_workers'])
    val_loader = torch.utils.data.DataLoader(full_dataset, batch_size=config['batch_size'], drop_last=True,
                                             sampler=valid_sampler, pin_memory=True, num_workers=config['num_workers'])

    # print(f' Train size: {len(train_dataset)},\n'
    #       f' Validation size: {len(val_dataset)},\n\n')

    return train_loader, val_loader


# test
if __name__ == '__main__':
    import sys
    data = sys.argv[1]

    config = {
        "data": data,
        "size": (512, 512),
        'random_seed': 42,
        "augments": False,
        "aug_str": "No_Augmentation",
        "data_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/',
        "result_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/results',
        "device": "cuda",
        "lr": 0.001,
        "batch_size": 4,
        "epochs": 60,
        "num_workers": 2,
    }

    train_loader, val_loader = get_loader(config)
    print(len(train_loader))
    print(len(val_loader))

    batch = iter(train_loader)
    images, labels = batch.next()

    print(images.shape)
    # torch.Size([num_samples, in_channels, H, W])

    print(labels.shape)
    # print(train_loader.dataset.shape)
