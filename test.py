import time
import numpy as np
from operator import add

from tqdm import tqdm
from dataset import get_data
from augmentaions import *
from models import *
from metrics import *
from utils import read_image, process_image


class Tester:
    def __init__(self, config):
        """

        :param model: String describing the model type. Can be DeepLab, TriUnet, ... TODO
        :param config: Contains hyperparameters : lr, epochs, batch_size, T_0, T_mult
        """
        self.config = config
        self.device = config["device"]
        self.loss_fn = config["loss_fn"]
        self.model_str = config["model"]
        self.data_str = config['data']
        self.result_path = config['result_path']
        self.augment_str = config['aug_str']
        self.predictor_name = f"{self.result_path}/{self.model_str}/{self.data_str}/{self.augment_str}_{self.loss_fn}.pth"
        # self.predictor_name = f"{self.result_path}/{self.model_str}/No_Augmentation_{self.loss_fn}-{self.data_str}"

        if self.model_str == "DeepLab":
            self.model = DeepLab().to(self.device)
            self.model.load_state_dict(torch.load(
                self.predictor_name, map_location=self.device))
        elif self.model_str == "TriUnet":
            self.model = TriUnet().to(self.device)
            self.model.load_state_dict(torch.load(
                self.predictor_name, map_location=self.device))
        elif self.model_str == "Unet":
            self.model = Unet().to(self.device)
            self.model.load_state_dict(torch.load(
                self.predictor_name, map_location=self.device))
        elif self.model_str == "Unet++":
            self.model = Unet(plusplus=True).to(self.device)
            self.model.load_state_dict(torch.load(
                self.predictor_name, map_location=self.device))
        elif self.model_str == "FPN":
            self.model = FPN().to(self.device)
            self.model.load_state_dict(torch.load(
                self.predictor_name, map_location=self.device))
        elif self.model_str == "MANet":
            self.model = MANet().to(self.device)
            self.model.load_state_dict(torch.load(
                self.predictor_name, map_location=self.device))

        else:
            raise AttributeError(
                "model_str not valid; choices are DeepLab, TriUnet, InductiveNet, FPN, Unet")

        # Load the data
        _, _, self.test_x, self.test_y = get_data(config)

    def test(self):

        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        time_taken = []
        for i, (x, y) in tqdm(enumerate(zip(self.test_x, self.test_y)), total=len(self.test_x)):

            """ read and process image """
            image, mask = read_image(x, y)
            
            x, y = process_image(image, mask)

            with torch.no_grad():
                """ Prediction and Calculating FPS """
                start_time = time.time()
                pred_y = self.model.predict(x)
                # pred_y = torch.sigmoid(pred_y)
                total_time = time.time() - start_time
                time_taken.append(total_time)

                score = calculate_metrics(y, pred_y)
                metrics_score = list(map(add, metrics_score, score))

        jaccard = metrics_score[0]/len(self.test_x)
        f1 = metrics_score[1]/len(self.test_x)
        recall = metrics_score[2]/len(self.test_x)
        precision = metrics_score[3]/len(self.test_x)
        acc = metrics_score[4]/len(self.test_x)
        roc_auc = metrics_score[5]/len(self.test_x)

        print(f"Results from {self.model_str} for {self.data_str} using {self.loss_fn}\n\n"
              f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f}\n"
              f"Precision: {precision:1.4f} - Acc: {acc:1.4f} - ROC-AUC: {roc_auc:1.4f}")

        fps = 1/np.mean(time_taken)
        print("FPS: ", fps)

        return metrics_score
