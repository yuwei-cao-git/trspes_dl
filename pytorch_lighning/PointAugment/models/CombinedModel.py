import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.opt_and_schedulars import get_optimizer_c, get_optimizer_a, get_lr_scheduler
from common.loss_utils import g_loss, d_loss, calc_loss
# from sklearn.metrics import r2_score
from torcheval.metrics.functional import r2_score
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
        self.class_weights = torch.Tensor(np.array(self.params["train_weights"]))
        self.num_classes = len(self.params["classes"])
        self.exp_name=self.params["exp_name"]
        self.best_test_loss = np.inf
        self.triggertimes = 0
        self.change = 0
        self.automatic_optimization=False
        self.best_test_outputs = None
        self.validation_step_outputs = []

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
        data, target = (data.cuda(), target.cuda().squeeze())
        data = data.permute(0, 2, 1)
        if self.use_augmentor:
            # Augmentor forward pass
            opt_c, opt_a = self.optimizers()

            noise = (0.02 * torch.randn(data.size(0), 1024)).cuda()
            # Classifier forward pass for both original and augmented data
            
            logits_data, logits_aug_data, aug_data = self(data, noise)

            # Compute losses
            loss_augmentor = g_loss(target, logits_data, logits_aug_data, data, aug_data, self.class_weights.cuda())
            
            self.manual_backward(loss_augmentor, retain_graph=True)

            loss_classifier = d_loss(target, logits_data, logits_aug_data, self.class_weights.cuda())
            self.manual_backward(loss_classifier)

            # Backward for augmentor
            opt_a.step()  # Update augmentor parameters
            opt_a.zero_grad()
            # Backward for classifier
            opt_c.step()
            opt_c.zero_grad()
            self.log_dict({"loss_classifier": loss_classifier, "loss_augmentor": loss_augmentor}, prog_bar=True)
            
            return {
                'class_loss': loss_classifier, 
                "aug_loss": loss_augmentor,  # Example output
                "data": data,  # Original point cloud
                "aug_data": aug_data,  # Augmented point cloud
            }
            
        else:
            opt_c = self.optimizers()
            # Classifier forward pass for original data only
            logits_data = self.classifier(data)
            
            # Compute loss
            loss_classifier = calc_loss(target, logits_data, self.class_weights.cuda())

            # Backward for classifier
            opt_c.zero_grad()
            self.manual_backward(loss_classifier)
            self.log('loss_classifier', loss_classifier, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            opt_c.step()  # Update classifier parametersv
            sch_c = self.lr_schedulers()
            self.lr_scheduler_step(sch_c, self.trainer.callback_metrics["val_loss"]) # TODO: should maunally step schedular?

            return {'class_loss': loss_classifier}  # Return loss for logging
    
    def on_train_epoch_end(self):
        # multiple schedulers
        sch_a, sch_c = self.lr_schedulers()
        self.lr_scheduler_step(sch_a, self.trainer.callback_metrics["val_loss"])
        self.lr_scheduler_step(sch_c, self.trainer.callback_metrics["val_loss"])
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        data = data.permute(0, 2, 1)
        # Forward pass
        logits_data = self(data)
        # Compute cross-entropy loss why not cross_entropy loss?
        loss = F.mse_loss(F.softmax(logits_data, dim=1), target)
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.validation_step_outputs.append({"val_loss": loss, "val_target": target, "val_pred": F.softmax(logits_data, dim=1)})
        self.log('val_loss', loss, on_step=True, prog_bar=True, logger=True) # TODO: on_step or on_epoch is needed

        return {'val_class_loss': loss}
    
    def on_validation_epoch_end(self): 
        last_epoch_val_loss = torch.mean(torch.stack([output['val_loss'] for output in self.validation_step_outputs]))
        
        test_true=torch.stack([output['val_target'] for output in self.validation_step_outputs])
        test_pred=torch.stack([output['val_pred'] for output in self.validation_step_outputs])
        rounded_pred=test_pred.flatten().round(decimals=2)
        val_r2 = r2_score(rounded_pred, test_true.flatten())
        self.log("val_r2", val_r2)

        self.log("ave_val_loss", last_epoch_val_loss, prog_bar=True)
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

    def configure_optimizers(self):
        optimizers = [get_optimizer_c(self.params, self.classifier)]
        schedulers = [get_lr_scheduler(self.params, optimizers[0], self.change)]

        if self.use_augmentor:
            optimizers.append(get_optimizer_a(self.params, self.augmentor))
            schedulers.append(get_lr_scheduler(self.params, optimizers[1], self.change))

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
                optimizer_configs.append(optimizer)
        
        return optimizer_configs

    def lr_scheduler_step(self, scheduler, metric):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        else:
            scheduler.step()
