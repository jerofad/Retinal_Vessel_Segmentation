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
               "size": (224, 224),
               "augments": False,
               "data_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/',
               "result_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/results',
               "device": "cuda",
               "lr": 0.05,
               "batch_size": 8,
               "epochs": 100,
               "num_workers": 2,
               }

    """
    Baseline Training
    """
    run_name = f"{configs['model']}_{configs['loss_fn']}_{configs['data']}"
    print(run_name)
    wandb.init(project="vessel segmentation final", config=configs,
               name=run_name, dir=configs["result_path"])
    trainer = Trainer(configs, sweep=False)
    trainer.train()

    # Run test script
    tester = Tester(configs)
    tester.test()

    # finish wand logging
    wandb.finish()
    """
    Model-based augmentations
    """
