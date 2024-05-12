from torch.optim import Adam, SGD

def get_optimizer(optimizer_name, model, lr, weight_decay):
    if optimizer_name == "Adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")