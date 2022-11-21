from test import Tester
from train import *
from test import *
import sys

if __name__ == '__main__':
    model = sys.argv[1]
    data = sys.argv[2]
    loss = sys.argv[3]

    configs = {"model": model,
               "data": data,
               "loss_fn": loss,
               "data_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/',
               "result_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/results/',
               "device": "cuda",
               "lr": 0.0001,
               "batch_size": 8,
               "epochs": 5,
               "num_workers": 2,
               }

    """
    Baseline Training
    """
    run_name = f"{model}_{loss}_{data}"
    wandb.init(project="vessel segmentation", config=configs, name=run_name)
    trainer = Trainer(configs)
    trainer.train()

    # Run test script
    tester = Tester(configs)
    tester.test()

    # finish wand logging
    wandb.finish()
    """
    Model-based augmentations
    """
