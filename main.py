from train import *
import sys

if __name__ == '__main__':
    model = sys.argv[1]
    data = sys.argv[2]

    config = {  "model": model,
                "data":data,
                "data_path":'/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/datasets/',
                "result_path":'/mnt/qb/berens/users/jfadugba97/RetinaSegmentation/results/',
                "device": "cuda",
                "lr": 0.0001,
                "batch_size": 8,
                "epochs": 200,
                "num_workers":2,
             }


    """
    Baseline Training
    """
    trainer = Trainer(config)
    trainer.train()
    """
    Model-based augmentations
    """
    