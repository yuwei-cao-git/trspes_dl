import os
import sys
import warnings
import argparse
import torch
from torch.utils.data import Subset

from utils.tools import (
    IOStream,
    PointCloudsInPickle,
    _init_,
)
from utils.augmentation import AugmentPointCloudsInPickle
from utils.train_ddp import test, train

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='params to init the multi-gpu settings')
    
parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--batch_size', default=48, type=int, help='')

def main(params):
    # set up folder structure
    _init_(params["exp_name"])

    # initiate IOStream
    io = IOStream("checkpoints/" + params["exp_name"] + "/run.log")
    io.cprint(params["exp_name"])
    io.cprint(str(params))

    if params["cuda"]:
        io.cprint("Using GPU")
    else:
        io.cprint("Using CPU")

    # Load datasets
    train_data_path = os.path.join(params["train_path"], str(params["num_points"]))
    train_pickle = params["train_pickle"]
    trainset = PointCloudsInPickle(train_data_path, train_pickle)
    if params["augment"] == True:
        for i in range(params["n_augs"]):
            aug_trainset = AugmentPointCloudsInPickle(
                train_data_path, 
                train_pickle,
            )
            
            trainset = torch.utils.data.ConcatDataset(
                [trainset, aug_trainset]
            )
    trainset_idx = list(range(len(trainset)))
    rem = len(trainset_idx) % params["batch_size"]
    if rem <= 3:
        trainset_idx = trainset_idx[: len(trainset_idx) - rem]
        trainset = Subset(trainset, trainset_idx)

    test_data_path = os.path.join(params["test_path"], str(params["num_points"]))
    test_pickle = params["test_pickle"]
    testset = PointCloudsInPickle(test_data_path, test_pickle)

    if not params["eval"]:
        train(params, io, trainset, testset)
        torch.cuda.empty_cache()
    else:
        test(params, io, testset)
        
        
if __name__ == "__main__":
    n_samples = [1944, 5358, 2250, 2630, 3982, 2034, 347, 9569, 397]
    class_weights = [1/(100*n/11057) for n in n_samples]
    args = parser.parse_args()
    params = {
        "exp_name": "pn2_ddp_PA_7168_WEIGHTS_AUG2",  # experiment name
        "model": "pn2",  # model
        "augmentor": True,
        "batch_size": args.batch_size,  # batch size
        "train_weights": class_weights, # training weights
        "train_path": r"../../../data/rmf_laz/train",
        "train_pickle": r"../../../data/rmf_laz/train/plots_comp.pkl",
        "test_path": r"../../../data/rmf_laz/val",
        "test_pickle": r"../../../data/rmf_laz/val/plots_comp.pkl",
        "augment": False, # augment
        "n_augs": 2, # number of augmentations
        "classes": ['BF', 'BW', 'CE', 'LA', 'PT', 'PJ', 'PO', 'SB', 'SW'],  # classes
        "n_gpus": torch.cuda.device_count(),  # number of gpus
        "epochs": 100,  # total epochs
        "optimizer_a": "adam",  # augmentor optimizer,
        "optimizer_c": "adam",  # classifier optimizer
        "lr_a": 1e-4,  # augmentor learning rate
        "lr_c": 5e-4,  # classifier learning rate
        "adaptive_lr": True,  # adaptive learning rate
        "patience": 10,  # patience
        "step_size": 20,  # step size
        "momentum": 0.9,  # sgd momentum
        "num_points": 7168,  # number of points
        "dropout": 0.5,  # dropout rate
        "emb_dims": 1024,  # dimension of embeddings
        "k": 20,  # k nearest points
        "model_path": "",  # pretrained model path
        "cuda": True,  # use cuda
        "eval": False,  # run testing
        "init_method": args.init_method,
    }

    mn = params["exp_name"]
    
    #send_telegram(f"Starting {mn}")
    main(params)
    # try:
    #     main(params)
    # except Exception as e:
    #     print(str(e))
    #     send_telegram(str(e))
