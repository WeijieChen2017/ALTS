from torch.optim.lr_scheduler import CosineAnnealingLR

def get_scheduler(scheduler_name, optimizer, T_max, eta_min):
    if scheduler_name == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")