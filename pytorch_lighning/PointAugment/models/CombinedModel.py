import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.opt_and_schedulars import get_optimizer_c, get_optimizer_a, get_lr_scheduler, get_lr_scheduler_step
from common.loss_utils import g_loss, d_loss, calc_loss
import torchmetrics
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor

    
class CombinedModel(L.LightningModule):
    def __init__(self, classifier, augmentor, params):
        super(CombinedModel, self).__init__()
        self.params = params
        self.use_augmentor = self.params["augmentor"]
        if self.use_augmentor:
            self.augmentor = augmentor
        self.classifier = classifier
        self.num_classes = len(self.params["classes"])
        self.exp_name=self.params["exp_name"]
        self.best_test_loss = np.inf
        self.triggertimes = 0
        self.change = 0
        self.automatic_optimization=False
        self.best_test_outputs = None
        self.validation_step_outputs = []
        
        #Metrics
        #micro_recall = torchmetrics.Accuracy(average="micro")
        #macro_recall = torchmetrics.Accuracy(average="macro", num_classes=self.num_classes)
        r2 = torchmetrics.R2Score()

        self.metrics = torchmetrics.MetricCollection(
            {#"Micro Accuracy":micro_recall,
             #"Macro Accuracy":macro_recall,
             "R2 Accuracy": r2
             })

        #self.save_hyperparameters(ignore=["loss_weight"])

    def forward(self, x, noise=None):
        if noise != None:
            group = (x, noise)
            aug_data = self.augmentor(group)
            logits_data = self.classifier(x)
            logits_aug_data = self.classifier(aug_data)
            return logits_data, logits_aug_data, aug_data
        else:
            logits_data = self.classifier(x)
            return logits_data

    def training_step(self, batch, batch_idx):
        data, target = batch
        data, target = (data, target.squeeze())
        data = data.permute(0, 2, 1)
        class_weights = torch.tensor(self.params["train_weights"], device="cuda", dtype=torch.float)
        # Augmentor forward pass
        opt_c, opt_a = self.optimizers()

        noise = (0.02 * torch.randn(data.size(0), 1024, device="cuda"))
        # Classifier forward pass for both original and augmented data
        
        logits_data, logits_aug_data, aug_data = self(data, noise)

        # Compute losses
        loss_augmentor = g_loss(target, logits_data, logits_aug_data, data, aug_data, class_weights)
        train_r2 = torchmetrics.functional.r2_score(torch.round(F.softmax(logits_data, dim=1).flatten(), decimals=2), target.flatten())
        self.manual_backward(loss_augmentor, retain_graph=True)

        loss_classifier = d_loss(target, logits_data, logits_aug_data, class_weights)
        self.manual_backward(loss_classifier)
        self.log_dict({"loss_classifier": loss_classifier, "loss_augmentor": loss_augmentor, "train_r2": train_r2}, prog_bar=True, sync_dist=True)
        # Backward for augmentor
        opt_a.step()  # Update augmentor parameters
        opt_a.zero_grad()
        # Backward for classifier
        opt_c.step()
        opt_c.zero_grad()
        
        return {
            'class_loss': loss_classifier, 
            "aug_loss": loss_augmentor,  # Example output
            "data": data,  # Original point cloud
            "aug_data": aug_data,  # Augmented point cloud
        }
    
    def on_train_epoch_end(self):
        # Aggregate predictions and targets
        sch_a, sch_c = self.lr_schedulers()
        self.lr_scheduler_step(sch_a, self.trainer.callback_metrics["val_loss"])
        self.lr_scheduler_step(sch_c, self.trainer.callback_metrics["val_loss"])
        self.log('lr_aug', sch_a.optimizer.param_groups[0]['lr'])
        self.log('lr_cla', sch_c.optimizer.param_groups[0]['lr'])
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        data, target = (data, target.squeeze())
        data = data.permute(0, 2, 1)
        # Forward pass
        logits_data = self(data, None)
        print(target.shape)
        print(logits_data.shape)
        # Compute cross-entropy loss why not cross_entropy loss?
        preds = F.softmax(logits_data, dim=1)
        loss = F.mse_loss(preds, target)
        
        # logs metrics for each training_step
        self.validation_step_outputs.append({"val_loss": loss, "val_target": target, "val_pred": preds})
        self.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True) # TODO: on_step or on_epoch is needed
        return {'val_loss': loss}
    
    def on_validation_epoch_end(self): 
        last_epoch_val_loss = torch.mean(torch.stack([output['val_loss'] for output in self.validation_step_outputs]))
        self.log("ave_val_loss", last_epoch_val_loss, prog_bar=True, sync_dist=True)

        test_true = torch.cat([output['val_target'] for output in self.validation_step_outputs], dim=0)
        test_pred = torch.cat([output['val_pred'] for output in self.validation_step_outputs], dim=0)
        test_pred = torch.round(test_pred.flatten(), decimals=2)
        test_true = test_true.flatten()
        self.log("ave_val_r2", torchmetrics.functional.r2_score(test_pred, test_true), sync_dist=True)

        if last_epoch_val_loss > self.best_test_loss:
            self.triggertimes += 1
            if self.triggertimes > self.params["patience"]:
                self.change = 1
        else:
            self.best_test_loss = last_epoch_val_loss
            self.triggertimes = 0
            # Update best model if current validation metric is better
            self.best_model_state_dict = self.state_dict()  # Save current model state
            self.best_test_outputs = (test_true, test_pred)  # Concatenate predictions and targets

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        data, target = batch
        data, target = (data.cuda(), target.cuda().squeeze())
        data = data.permute(0, 2, 1)
        output = self.model(data)
        loss = F.mse_loss(F.softmax(output, dim=1), target)
        self.log("test_loss", loss, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizers = [get_optimizer_c(self.params, self.classifier)]
        schedulers = [get_lr_scheduler_step(self.params, optimizers[0])]

        if self.use_augmentor:
            optimizers.append(get_optimizer_a(self.params, self.augmentor))
            schedulers.append(get_lr_scheduler_step(self.params, optimizers[1]))

        # Build the configuration
        optimizer_configs = []
        for optimizer, scheduler in zip(optimizers, schedulers):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                optimizer_configs.append({
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val_loss'  # Adjust to the correct metric you want to monitor
                    }
                })
            else:
                optimizer_configs.append({
                    'optimizer': optimizer,
                    'lr_scheduler': scheduler
                })
        
        return optimizer_configs

    def lr_scheduler_step(self, scheduler, metric):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        else:
            scheduler.step()
