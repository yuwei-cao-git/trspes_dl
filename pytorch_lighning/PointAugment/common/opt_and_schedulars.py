from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def get_optimizer_c(params, model):
    if params['optimizer_c'] == "Adam":
        optimizer = Adam(model.parameters(),
                                     lr=params['lr_c'],
                                     betas=(0.9, 0.999), eps=1e-08)
    else:
        optimizer = SGD(params=model.parameters(),
                                    lr=params['lr_c'],
                                    momentum=params["momentum"],
                                    weight_decay=1e-4)

    return optimizer

def get_optimizer_a(params, model):
    if params['optimizer_c'] == "Adam":
        optimizer = Adam(model.parameters(),
                        lr=params['lr_a'],
                        betas=(0.9, 0.999), eps=1e-08)
    else:
        optimizer = SGD(params=model.parameters(),
                        lr=params['lr_a'],
                        momentum=params["momentum"],
                        weight_decay=1e-4)

    return optimizer


def get_lr_scheduler(params, optimizer, change):
    if params["adaptive_lr"]:
        if change == 1:
            scheduler = StepLR(optimizer,
                            step_size=params['step_size'],
                            gamma=0.1)
        else:
            scheduler = ReduceLROnPlateau(optimizer,
                            mode='min',
                            patience=params['patience']) # Number of epochs with no improvement after which learning rate will be reduced.
        return scheduler
    else:
        return None