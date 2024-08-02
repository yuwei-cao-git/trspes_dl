import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import Subset
import random
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
import wandb
from torch.utils.data import DataLoader
from models.augmentor import Augmentor
from models.dgcnn import DGCNN
from models.CombinedModel import CombinedModel
from utils.tools import (
    IOStream,
    PointCloudsInPickle,
    _init_,
)
from utils.tools import create_comp_csv, write_las
from utils.augmentation import AugmentPointCloudsInPickle
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser(description="pytorch-lightning parallel test")
parser.add_argument("--lr", type=float, default=0.1, help="")
parser.add_argument("--max_epochs", type=int, default=4, help="")
parser.add_argument("--batch_size", type=int, default=4, help="")
parser.add_argument("--num_workers", type=int, default=8, help="")


def prepare_dataset(params):
    # Load datasets
    train_data_path = os.path.join(params["train_path"], str(params["num_points"]))
    train_pickle = params["train_pickle"]
    trainset = PointCloudsInPickle(train_data_path, train_pickle)
    trainset_idx = list(range(len(trainset)))
    rem = len(trainset_idx) % params["batch_size"]
    if rem <= 3:
        trainset_idx = trainset_idx[: len(trainset_idx) - rem]
        trainset = Subset(trainset, trainset_idx)
    if params["augment"] == True:
        for i in range(params["n_augs"]):
            aug_trainset = AugmentPointCloudsInPickle(
                train_data_path,
                train_pickle,
            )

            trainset = torch.utils.data.ConcatDataset([trainset, aug_trainset])
    train_loader = DataLoader(
        trainset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        sampler=None,
        collate_fn=None,
        drop_last=True,
        pin_memory=True,
    )

    test_data_path = os.path.join(params["test_path"], str(params["num_points"]))
    test_pickle = params["test_pickle"]
    testset = PointCloudsInPickle(test_data_path, test_pickle)
    test_loader = DataLoader(
        testset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        sampler=None,
        collate_fn=None,
        drop_last=True,
        pin_memory=True,
    )

    return train_loader, test_loader


class PointCloudLogger(Callback):
    def __init__(self, exp_name, write_las_function):
        self.exp_name = exp_name
        self.write_las = (
            write_las_function  # Assuming write_las is a function to write LAZ files
        )
        self.log_epochs = [1, 50, 100, 150, 200]  # Epochs to log

    def on_batch_end(self, trainer, batch, batch_idx, pl_module, dataloader_idx):
        # Access data and potentially augmented point cloud from current batch
        data = batch["data"]
        aug_data = batch["aug_data"]
        if (
            random.random() > 0.99
        ):  # Log 0.01 the batches at specified epochs (adjust as needed)
            true_pc_np = data.detach().cpu().numpy()
            self.write_las(
                true_pc_np[1],
                f"checkpoints/{self.exp_name}/output/laz/epoch{trainer.current_epoch}_pc{batch_idx}_true.laz",
            )
            wandb.log({"point_cloud": wandb.Object3D(true_pc_np)})
            if aug_data is not None:  # Handle logging augmented data if applicable
                aug_data_np = aug_data.detach().cpu().numpy()
                self.write_las(
                    aug_data_np[1],
                    f"checkpoints/{self.exp_name}/output/laz/epoch{trainer.current_epoch}_pc{batch_idx}_aug.laz",
                )
                wandb.log({"point_cloud": wandb.Object3D(aug_data_np)})


def main(params):
    print("Starting...")
    L.seed_everything(1)
    # Initialize WandB, CSV Loggers
    wandb_logger = WandbLogger(project="tree_species_composition_dl_pl")
    exp_name = params["exp_name"]
    exp_dirpath = os.path.join("checkpoints", exp_name)
    output_dir = Path(os.path.join(exp_dirpath, "output"))
    output_dir.mkdir(parents=True, exist_ok=True) 
    csv_logger = CSVLogger(save_dir=output_dir, name="loss_r2")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(exp_dirpath, "models"),  # Path to save checkpoints
        filename="{epoch}-{val_loss:.2f}",  # Filename format (epoch-val_loss)
        monitor="val_loss",  # Metric to monitor for "best" model (can be any logged metric)
        mode="min",  # Save model with the lowest "val_loss" (change to "max" for maximizing metrics)
        save_top_k=1,  # Save only the single best model based on the monitored metric
    )
    pointcloud_logger = PointCloudLogger(
        exp_name=exp_name, write_las_function=write_las
    )

    # initialize model
    augmentor = Augmentor()
    classifier = DGCNN(params, len(params["classes"]))
    model = CombinedModel(classifier, augmentor, params)

    train_dataloader, val_dataloader = prepare_dataset(params)
    # ddp = DDPStrategy(process_group_backend="nccl")
    # Instantiate the Trainer
    trainer = Trainer(
        max_epochs=params["epochs"],
        logger=[wandb_logger, csv_logger],  # csv_logger
        callbacks=[checkpoint_callback, pointcloud_logger],
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    if model.best_test_outputs is not None:
        test_true, test_pred = model.best_test_outputs
        test_pred=test_pred.detach().cpu().numpy()
        test_true=test_true.detach().cpu().numpy()
        r2_score=r2_score(test_true,test_pred)
        wandb.log({"sk_r2": r2_score})
        create_comp_csv(
            test_true,
            test_pred,
            params["classes"],
            f"checkpoints/{exp_name}/output/best_model_outputs.csv",
        )
    if params["eval"]:
        trainer.test(model, val_dataloader)


if __name__ == "__main__":
    n_samples = [1944, 5358, 2250, 2630, 3982, 2034, 347, 9569, 397]
    class_weights = [1 / (100 * n / 11057) for n in n_samples]
    args = parser.parse_args()
    params = {
        "exp_name": "DGCNN_pointaugment_7168_WEIGHTS_AUG2",  # experiment name
        "augmentor": True,
        "batch_size": args.batch_size,  # batch size
        "train_weights": class_weights,  # training weights
        "train_path": r"../../data/rmf_laz/train",
        "train_pickle": r"../../data/rmf_laz/train/plots_comp.pkl",
        "test_path": r"../../data/rmf_laz/val",
        "test_pickle": r"../../data/rmf_laz/val/plots_comp.pkl",
        "augment": True,  # augment
        "n_augs": 2,  # number of augmentations
        "classes": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],  # classes
        "n_gpus": torch.cuda.device_count(),  # number of gpus
        "epochs": args.max_epochs,  # total epochs
        "optimizer_a": "adam",  # augmentor optimizer,
        "optimizer_c": "adam",  # classifier optimizer
        "lr_a": args.lr,  # augmentor learning rate
        "lr_c": args.lr,  # classifier learning rate
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
        "num_workers": args.num_workers,  # num_cpu_per_gpu
    }

    mn = params["exp_name"]
    main(params)
