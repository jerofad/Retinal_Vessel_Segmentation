from test import Tester
from train import *
from test import *
import sys


default = {"model": "Unet",
           "data": "fives",
           "loss_fn": "DiceLoss",
           "size": (224, 224),
           "augments": False,
           "data_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/',
           "result_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/results',
           "device": "cuda",
           "learning_rate": 0.05,
           "batch_size": 8,
           "epochs": 50,
           "num_workers": 2,
           }


def main():

    wandb.init(project='sweep_dice',
               config=default, dir=default["result_path"])
    config = wandb.config
    trainer = Trainer(config, sweep=True)
    trainer.train()


# 🐝 Step 2: Define sweep config
sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'minimize',
               'name': 'val_loss'
               },
    'parameters':
    {
        'batch_size': {'values': [4, 8]},
        'learning_rate': {'values': [0.1, 0.01, 0.001, 0.0001, 0.00001]}
    }
}

# 🐝 Step 3: Initialize sweep by passing in config
sweep_id = wandb.sweep(sweep=sweep_configuration,
                      project='sweep_dice')

# 🐝 Step 4: Call to `wandb.agent` to start a sweep
wandb.agent(sweep_id, function=main)
