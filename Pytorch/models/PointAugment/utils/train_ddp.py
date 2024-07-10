import os
import random
import warnings
import time
import datetime
from pathlib import Path
import wandb

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from augment.augmentor import Augmentor
from models.dgcnn import DGCNN
from models.pointnet2 import PointNet2
from common import loss_utils
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.tools import create_comp_csv, delete_files, variable_df, write_las, plot_3d, plot_2d
# import torch.profiler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

#warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
wandb.login()

def print_trainable_variables(model):
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(f"Variable: {name}, Device: {param.device}, Version{param._version}")

def get_resources(verbose=True):

    rank = 0
    local_rank = 0
    world_size = 1
    ngpus_per_node = torch.cuda.device_count()

    # launched with srun (SLURM)
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank
    world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])*int(os.environ["SLURM_JOB_NUM_NODES"])
    local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    if verbose and rank == 0:
        print("launch with srun")

    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])

    return rank, local_rank, world_size, local_size, num_workers

def train(params, io, trainset, testset):
    
    # log using wandb
    runs = wandb.init(
        project="trspes_dl-Pytorch_models_PointAugment",
        #group=f'group-{rank}',
        #settings=wandb.Settings(start_method="fork"),
        config={
            "model": params["model"],
            "init_learning_rate_a": params["lr_a"],
            "inlearning_rate_c": params["lr_c"],
            "epoch": params["epochs"],
            "batch size": params["batch_size"],
        },
    )
    # parallel settings
    rank, local_rank, world_size, _, num_workers = get_resources()
    current_device = local_rank
    torch.cuda.set_device(current_device)

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    dist.init_process_group("nccl", init_method=params["init_method"], world_size=world_size, rank=rank)
    print("process group ready!")
    print('From Rank: {}, ==> Making model..'.format(rank))

    # Classifier
    if params["model"] == "dgcnn":
        classifier = DGCNN(params, len(params["classes"])).cuda()
    elif params["model"] == "pn2":
        classifier = PointNet2(len(params["classes"])).cuda()
    else:
        raise Exception("Model Not Implemented")
    # Wrap the model with DDP
    classifier = DDP(classifier, device_ids=[current_device], output_device=local_rank)
    # Augmentor
    if params["augmentor"]:
        augmentor = Augmentor().cuda()
        augmentor = DDP(augmentor, device_ids=[current_device], output_device=local_rank)
   
    # model parameters
    exp_name = params["exp_name"]

    # Set up optimizers
    if params["augmentor"]:
        if params["optimizer_a"] == "sgd":
            optimizer_a = optim.SGD(
                augmentor.parameters(),
                lr=params["lr_a"],
                momentum=params["momentum"],
                weight_decay=1e-4,
            )
        elif params["optimizer_a"] == "adam":
            optimizer_a = optim.Adam(
                augmentor.parameters(), lr=params["lr_a"], betas=(0.9, 0.999), eps=1e-08
            )
        else:
            raise Exception("Optimizer Not Implemented")

    if params["optimizer_c"] == "sgd":
        optimizer_c = optim.SGD(
            classifier.parameters(),
            lr=params["lr_c"],
            momentum=params["momentum"],
            weight_decay=1e-4,
        )
    elif params["optimizer_c"] == "adam":
        optimizer_c = optim.Adam(
            classifier.parameters(), lr=params["lr_c"], betas=(0.9, 0.999), eps=1e-08
        )
    else:
        raise Exception("Optimizer Not Implemented")

    # Adaptive Learning
    if params["adaptive_lr"] is True:
        scheduler1_c = ReduceLROnPlateau(optimizer_c, "min", patience=params["patience"])
        scheduler2_c = StepLR(optimizer_c, step_size=params["step_size"], gamma=0.1)
        if params["augmentor"]:
            scheduler1_a = ReduceLROnPlateau(optimizer_a, "min", patience=params["patience"])
            scheduler2_a = StepLR(optimizer_a, step_size=params["step_size"], gamma=0.1)
        change = 0

    # Set initial best test loss
    best_test_loss = np.inf

    # Set initial triggertimes
    triggertimes = 0
    
    weights = params["train_weights"]
    weights = torch.Tensor(np.array(weights)).cuda()

    if rank == 0:
        wandb.alert(title="training status", text="start training")
    tic = time.perf_counter()

    print('From Rank: {}, ==> Preparing data..'.format(rank))
    # distribute data
    train_sampler = DistributedSampler(dataset=trainset)
    test_sampler = DistributedSampler(dataset=testset)
    train_loader = DataLoader(trainset, batch_size=params["batch_size"], shuffle=(train_sampler is None), num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(testset, batch_size=params["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True, sampler=test_sampler)

    # Iterate through number of epochs
    for epoch in tqdm(range(params["epochs"]), desc="Model Total: ", leave=False, colour="red"):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        train_loss_a = 0.0
        train_loss_c = 0.0
        count = 0
        true_pred = []
        aug_pred = []
        train_true = []
        j=0
        if rank == 0:
            wandb.log({"epoch": epoch+1})
        # load data
        for data, label in tqdm(
            train_loader, desc="Training Total: ", leave=False, colour="cyan"
        ):
            start = time.time()

            # Get data and label, move data and label to the same device as model
            data, label = (data.cuda(), label.cuda().squeeze())

            # Permute data into correct shape
            data = data.permute(0, 2, 1)  # adapt augmentor to fit with this permutation

            # Get batch size
            batch_size = data.size()[0]
            classifier.train()
            optimizer_c.zero_grad()  # zero gradients

            # Augment
            if params["augmentor"]:
                noise = (0.02 * torch.randn(batch_size, 1024))
                noise = noise.cuda()
                group = (data, noise)
                augmentor.train()
                optimizer_a.zero_grad()  # zero gradients
                aug_pc = augmentor(group)

            # augmentor Loss
            out_true = classifier(data)  # classify truth
            if params["augmentor"]:
                out_aug = classifier(aug_pc)  # classify augmented
                cls_loss = loss_utils.d_loss(label, out_true, out_aug.detach(), weights)
            else:
                cls_loss = loss_utils.calc_loss(label, out_true, weights)
            cls_loss.backward(retain_graph=True)
            optimizer_c.step()

            # Augmentor Loss
            if params["augmentor"]:
                aug_loss = loss_utils.g_loss(label, out_true, out_aug, data, aug_pc, weights)
                # Backward
                aug_loss.backward(retain_graph=True)
                optimizer_a.step()
            
            # Update loss' and count
            if params["augmentor"]:
                train_loss_a += aug_loss.item()
            train_loss_c += cls_loss.item()
            count = batch_size
            
            # Append true/pred
            if rank==0:
                label_np = label.cpu().numpy()
                if label_np.ndim == 2:
                    train_true.append(label_np)
                else:
                    label_np = label_np[np.newaxis, :]
                    train_true.append(label_np)
                
                if params["augmentor"]:
                    aug_np = F.softmax(out_aug, dim=1)
                    aug_np = aug_np.detach().cpu().numpy()
                    if aug_np.ndim == 2:
                        aug_pred.append(aug_np)
                    else:
                        aug_np = aug_np[np.newaxis, :]
                        aug_pred.append(aug_np)
                
                true_np = F.softmax(out_true, dim=1)
                true_np = true_np.detach().cpu().numpy()
                if true_np.ndim ==2:
                    true_pred.append(true_np)
                else:
                    true_np = true_np[np.newaxis, :]
                    true_pred.append(true_np)
                if params["augmentor"]:
                    if epoch + 1 in [1, 50, 100, 150, 200, 250, 300]:
                        if rank==0:
                            if random.random() > 0.99:
                                aug_pc_np = aug_pc.detach().cpu().numpy()
                                true_pc_np = data.detach().cpu().numpy()
                                try:
                                    write_las(aug_pc_np[1], f"checkpoints/{exp_name}/output/laz/epoch{epoch + 1}_pc{j}_aug.laz")
                                    write_las(true_pc_np[1], f"checkpoints/{exp_name}/output/laz/epoch{epoch + 1}_pc{j}_true.laz")
                                    j+=1
                                except:
                                    j+=1
            batch_time = time.time() - start
            elapse_time = time.time() - epoch_start
            elapse_time = datetime.timedelta(seconds=elapse_time)
            #io.cprint("From Rank: {}, Training time {}".format(rank, elapse_time))
            wandb.log({
                    "rank": rank,
                    "batch_time": batch_time
            })

        # Concatenate true/pred
        train_true = np.concatenate(train_true)
        if params["augmentor"]:
            aug_pred = np.concatenate(aug_pred)
        true_pred = np.concatenate(true_pred)
        
        # Calculate R2's
        if params["augmentor"]:
            aug_r2 = r2_score(train_true.flatten(), aug_pred.flatten().round(2))
        true_r2 = r2_score(train_true.flatten(), true_pred.flatten().round(2))
        
        if params["augmentor"]:
            train_r2 = float(aug_r2 + true_r2) / 2
        else:
            train_r2 = true_r2
        wandb.log({"train_r2": train_r2})

        # Get average loss'
        if params["augmentor"]:
            train_loss_a = float(train_loss_a) / count
            wandb.log({"aug_loss": train_loss_a})
        train_loss_c = float(train_loss_c) / count
        wandb.log({"class_loss": train_loss_c})
        
        # Set up Validation
        classifier.eval()
        with torch.no_grad():
            test_sampler.set_epoch(epoch)
            test_loss = 0.0
            count = 0
            test_pred = []
            test_true = []

            # Validation
            for data, label in tqdm(
                test_loader, desc="Validation Total: ", leave=False, colour="green"
            ):
                # Get data and label
                data, label = (data.cuda(), label.cuda().squeeze())

                # Permute data into correct shape
                data = data.permute(0, 2, 1)

                # Get batch size
                batch_size = data.size()[0]

                # Run model
                output = classifier(data)

                # Calculate loss
                loss = F.mse_loss(F.softmax(output, dim=1), target=label)

                # Update count and test_loss
                count += batch_size
                test_loss += loss.item()
                if rank == 0:
                    # Append true/pred
                    label_np = label.cpu().numpy()
                    if label_np.ndim == 2:
                        test_true.append(label_np)
                    else:
                        label_np = label_np[np.newaxis, :]
                        test_true.append(label_np)

                    pred_np = F.softmax(output, dim=1)
                    pred_np = pred_np.detach().cpu().numpy()
                    if pred_np.ndim == 2:
                        test_pred.append(pred_np)
                    else:
                        pred_np = pred_np[np.newaxis, :]
                        test_pred.append(pred_np)
                
            # Concatenate true/pred
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)

            # Calculate R2
            val_r2 = r2_score(test_true.flatten(), test_pred.flatten().round(2))
            wandb.log({"val_r2": val_r2})
            # get average test loss
            test_loss = float(test_loss) / count
            wandb.log({"val_loss": test_loss})

        if rank == 0:
            # print and save losses and r2
            if params["augmentor"]:
                io.cprint(f"Epoch: {epoch + 1}, Training - Augmentor Loss: {train_loss_a}, Training - Classifier Loss: {train_loss_c}, Training R2: {train_r2}, Validation Loss: {test_loss}, R2: {val_r2}, Epoch time: {time.time() - epoch_start}")

                # Create output Dataframe
                out_dict = {"epoch": [epoch + 1],
                            "aug_loss": [train_loss_a],
                            "class_loss": [train_loss_c],
                            "train_r2": [train_r2],
                            "val_loss": [test_loss],
                            "val_r2": [val_r2]}
                wandb.log({"aug_loss": train_loss_a})
            else:
                io.cprint(f"Epoch: {epoch + 1}, Training - Classifier Loss: {train_loss_c}, Training R2: {train_r2}, Validation Loss: {test_loss}, R2: {val_r2}, Epoch time: {time.time() - epoch_start}")
                # Create output Dataframe
                out_dict = {"epoch": [epoch + 1],
                            "class_loss": [train_loss_c],
                            "train_r2": [train_r2],
                            "val_loss": [test_loss],
                            "val_r2": [val_r2]}
                
            out_df = pd.DataFrame.from_dict(out_dict)
            
            if not Path(f"checkpoints/{exp_name}/loss_r2.csv").exists:
                loss_r2_df = pd.read_csv(f"checkpoints/{exp_name}/loss_r2.csv")
                loss_r2_df = pd.concat([loss_r2_df, out_df])
                loss_r2_df.to_csv(f"checkpoints/{exp_name}/loss_r2.csv", index=False)
            else:
                out_df.to_csv(f"checkpoints/{exp_name}/loss_r2.csv", index=False)
            
            # Save Best Model # move after apply addaptive learning, as if put it front, the best_test_loss=test loss, which is never excute the step
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                if params["augmentor"]:
                    io.cprint(f"Training - Augmentor Loss: {train_loss_a}, Training - Classifier Loss: {train_loss_c}, Training R2: {train_r2}, Validation Loss: {test_loss}, R2: {val_r2}")
                else:
                    io.cprint(f"Training - Classifier Loss: {train_loss_c}, Training R2: {train_r2}, Best validation Loss: {test_loss}, R2: {val_r2}")
                    # wandb.alert(title="training status", text="archive better result!")
                
                torch.save(
                    classifier.state_dict(), f"checkpoints/{exp_name}/models/best_mode.t7"
                )

                # delete old files
                delete_files(f"checkpoints/{exp_name}/output", "*.csv")

                # Create CSV of best model output
                create_comp_csv(
                    test_true.flatten(),
                    test_pred.round(2).flatten(),
                    params["classes"],
                    f"checkpoints/{exp_name}/output/outputs_epoch{epoch+1}.csv",
                )
            
        # Apply addaptive learning
        if params["adaptive_lr"] is True:
            if test_loss > best_test_loss:
                triggertimes += 1
                if triggertimes > params["patience"]:
                    change = 1
            else:
                triggertimes = 0
            if change == 0:
                if params["augmentor"]:
                    scheduler1_a.step(test_loss)
                scheduler1_c.step(test_loss)
                if rank == 0:
                    if params["augmentor"]:
                        wandb.log({
                            "Scheduler Plateau Augmentor LR": scheduler1_a.optimizer.param_groups[0]['lr'],
                            "Trigger Times": triggertimes,
                        })
                    wandb.log({
                            "Scheduler Plateau Classifier LR": scheduler1_c.optimizer.param_groups[0]['lr'],
                            "Trigger Times": triggertimes,
                    })
            else:
                if params["augmentor"]:
                    scheduler2_a.step()
                scheduler2_c.step()
                if rank == 0:
                    if params["augmentor"]:
                        wandb.log({
                            "Scheduler Step Augmentor LR": scheduler2_a.optimizer.param_groups[0]['lr'],
                        })
                    wandb.log({
                        "Scheduler Step Classifier LR": scheduler2_c.optimizer.param_groups[0]['lr'],

                    })

        epoch_time = time.time() - epoch_start
        if rank == 0:
            wandb.log({"epoch_time": epoch_time})
    tac = time.perf_counter()
    if rank == 0:
        wandb.alert(title="training status", text="training end!")
        wandb.log({"Total Time": tac-tic})
    # clean up
    wandb.finish()
    dist.destroy_process_group()                          

def test(params, io, testset):
    device = torch.device("cuda" if params["cuda"] else "cpu")

    # Load model
    if params["model"] == "dgcnn":
        model = DGCNN(params, len(params["classes"])).to(device)
    else:
        raise Exception("Model Not Implemented")
        
    # Data Parallel
    model = nn.DataParallel(model, device_ids=list(range(0, params["n_gpus"])))

    # Load Pretrained Model
    model.load_state_dict(torch.load(params["model_path"]))
    
    # Setup for Testing
    model = model.eval()
    test_true = []
    test_pred = []
    test_loader = DataLoader(testset, batch_size=params["batch_size"], shuffle=False, pin_memory=True)
                                 
    # Testing
    for data, label in tqdm(
        test_loader, desc="Testing Total: ", leave=False, colour="green"
    ):
        # Get data, labels, & batch size
        data, label = (
            data.to(device),
            label.to(device).squeeze(),
        )
        data = data.permute(0, 2, 1)
        #batch_size = data.size()[0]

        # Run Model
        output = model(data)

        # Append true/pred
        test_true.append(label.cpu().numpy())
        test_pred.append(output.detach().cpu().numpy())

    # Concatenate true/pred
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    # Calculate R2
    r2 = r2_score(test_true, test_pred.round(2))

    io.cprint(f"R2: {r2}")
    wandb.log({"R2": r2})