import os
import numpy as np
from operator import add
import torch.optim.optimizer
from torch.utils.data import DataLoader
from dataset import get_loader
from tqdm import tqdm
from loss.dice import DiceBCELoss
from loss.soft_cldice import Soft_Dice_clDice
from loss.topo_loss import TopoLoss
from augmentaions import *
from models import *
from metrics import *
import wandb




class Trainer:
    def __init__(self, config, sweep=False):
        """

        :param model: String describing the model type. Can be DeepLab, TriUnet, ... TODO
        :param config: Contains hyperparameters : lr, epochs, batch_size, T_0, T_mult
        """
        self.config = config
        self.sweep = sweep
        self.device = config["device"]
        # We are tuning these parameters so we get them from wandb config
        self.lr = wandb.config.learning_rate #config["learning_rate"]
        self.batch_size = wandb.config.batch_size
        self.epochs = wandb.config.epochs
        
        self.model = None
        self.loss_fn = config["loss_fn"]
        self.model_str = config["model"]
        self.data_str = config['data']
        self.result_path = config['result_path']
        self.predictor_name = f"{self.result_path}/{self.model_str}/{self.data_str}/No_Augmentation_{self.loss_fn}"
        if self.sweep:
            self.predictor_name = f"{self.result_path}/{self.model_str}/{self.data_str}/No_Augmentation{self.loss_fn}_sweep"

        # create directory if doens not exists.
        self.model_dir = f"{self.result_path}/{self.model_str}/{self.data_str}"

        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if self.model_str == "DeepLab":
            self.model = DeepLab().to(self.device)
        elif self.model_str == "TriUnet":
            self.model = TriUnet().to(self.device)
        elif self.model_str == "Unet":
            self.model = Unet().to(self.device)
        elif self.model_str == "Unet++":
            self.model = Unet(plusplus=True).to(self.device)
        elif self.model_str == "FPN":
            self.model = FPN().to(self.device)
        elif self.model_str == "MANet":
            self.model = MANet().to(self.device)
        else:
            raise AttributeError(
                "model_str not valid; choices are DeepLab, TriUnet, InductiveNet, FPN, Unet")

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        if self.loss_fn == "DiceLoss":
            self.criterion = DiceBCELoss()
        elif self.loss_fn == "clDice":
            self.criterion = Soft_Dice_clDice()
        elif self.loss_fn == "topoloss":
            self.criterion = TopoLoss()
        else:
            raise AttributeError(
                f"Loss function {self.loss_fn} not valid: Choises are DiceLoss, clDice, topoloss")

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                           patience=5, verbose=True)
#         self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
#             self.optimizer, milestones=[30, 60, 90], gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer, T_0=50, T_mult=2)
        self.train_loader, self.val_loader = get_loader(self.config)

    def train_epoch(self):
        self.model.train()
        # Track model gradient and parameters.
        wandb.watch(self.model, self.criterion, log="all")

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
        # print(f"Starting {self.model_str} Segmentation training")

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
            wandb.log({'epoch': i,
                       'training_loss': training_loss,
                       'val_loss': val_loss,
                       'val_iou': mean_iou})
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


# This is used if we want to run the sweep
# if __name__ == '__main__':


#     default = {"model": "Unet",
#                "data": "fives",
#                "loss_fn": "DiceLoss",
#                "size": (224, 224),
#                "augments": False,
#                "data_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/',
#                "result_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/results',
#                "device": "cuda",
#                "learning_rate": 0.05,
#                "batch_size": 8,
#                "epochs": 100,
#                "num_workers": 2,
#                }

#     wandb.init(config=default, project="sweep_dice")
#     config = wandb.config
#     trainer = Trainer(config, sweep=True)
#     trainer.train()