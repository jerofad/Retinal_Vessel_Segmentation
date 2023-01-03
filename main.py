
from test import Tester
from train import Trainer
import wandb
import sys


if __name__ == '__main__':
    model = sys.argv[1]
    data = sys.argv[2]
    loss = sys.argv[3]

    configs = {"model": model,
               "data": data,
               "loss_fn": loss,
               "size": (512, 512),
               'random_seed':42,
               "augments": False,
               "aug_str":"Augmentation",
               "data_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/',
               "result_path": '/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/results',
               "device": "cuda",
               "learning_rate": 0.01,
               "batch_size": 4,
               "epochs": 50,
               "num_workers": 2,
               }

    """
    Baseline Training
    """
    run_name = f"{configs['model']}_{configs['loss_fn']}_{configs['data']}"
    print(run_name)
    wandb.init(project="RVS", config=configs,
               name=run_name, dir=configs["result_path"])

    trainer = Trainer(configs, sweep=False)
    trainer.train()

    # # Run test script
    # tester = Tester(configs)
    # tester.test()

    # finish wand logging
    wandb.finish()
    """
    Model-based augmentations
    """
