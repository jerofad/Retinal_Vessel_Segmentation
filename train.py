
import numpy as np
from operator import add
import torch.optim.optimizer
from torch.utils.data import DataLoader
from dataset import get_loader
from tqdm import tqdm
from loss import DiceBCELoss
from augmentaions import *
from models import *
from metrics import *

class Trainer:
    def __init__(self, config):
        """

        :param model: String describing the model type. Can be DeepLab, TriUnet, ... TODO
        :param config: Contains hyperparameters : lr, epochs, batch_size, T_0, T_mult
        """
        self.config = config
        self.device = config["device"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.model = None
        self.model_str = config["model"]
        self.data_str = config['data']
        self.result_path = config['result_path']
        self.mnv = Augmentations()
        #TODO: change model_path name
        self.predictor_name = f"{self.result_path}/No_Augmentation_{self.model_str}-{self.data_str}"
   
        if self.model_str == "DeepLab":
            self.model = DeepLab().to(self.device)
        elif self.model_str == "TriUnet":
            self.model = TriUnet().to(self.device)
        elif self.model_str == "Unet":
            self.model = Unet().to(self.device)
        elif self.model_str == "FPN":
            self.model = FPN().to(self.device)
        else:
            raise AttributeError("model_str not valid; choices are DeepLab, TriUnet, InductiveNet, FPN, Unet")

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = DiceBCELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2)
        self.train_loader, self.val_loader = get_loader(self.config)


    def train_epoch(self):
        self.model.train()
        losses = []
        for i, (x, y) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            image = x.to(self.device)
            mask = y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, mask)
            loss.backward()
            self.optimizer.step()
            losses.append(np.abs(loss.item()))
        return np.mean(losses)

    def train(self):
        best_val_loss = 10
        print(f"Starting {self.model_str} Segmentation training")

        for i in range(self.epochs):
            training_loss = np.abs(self.train_epoch())
            val_loss, ious = self.validate(epoch=i, plot=False)

            mean_iou = float(torch.mean(ious))
            self.scheduler.step(i)
            print(
                f"Epoch {i} of {self.epochs} \t"
                f" lr={[group['lr'] for group in self.optimizer.param_groups]} \t"
                f" loss={training_loss} \t"
                f" val_loss={val_loss} \t"
                f" val_iou={mean_iou} \t"
            )
            if val_loss < best_val_loss:
                data_str = f"Saving new best model. Valid loss improved from {best_val_loss:2.4f} to {val_loss:2.4f}"
                print(data_str)
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.predictor_name)


    def validate(self, epoch, plot=False):
        self.model.eval()
        losses = []
        ious = torch.empty((0,)).to(self.device)
        with torch.no_grad():
            for x, y in self.val_loader:
                image = x.to(self.device)
                mask = y.to(self.device)
                # aug_img, aug_mask = self.mnv(image, mask)
                output = self.model(image)
                # aug_output = self.model(aug_img)

                batch_ious = torch.mean(iou(output, mask))
                loss = self.criterion(output, mask)
                losses.append(np.abs(loss.item()))
                ious = torch.cat((ious, batch_ious.flatten()))
        avg_val_loss = np.mean(losses)
        return avg_val_loss, ious
