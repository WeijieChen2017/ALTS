# from monai import loss

from monai.losses import DiceLoss, CrossEntropyLoss, FocalLoss, GeneralizedDiceLoss, TverskyLoss

def get_loss(loss_name: str):
    if loss_name == "DiceLoss":
        return loss.DiceLoss(sigmoid=True, squared_pred=True)
    elif loss_name == "CrossEntropyLoss":
        return loss.CrossEntropyLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "FocalLoss":
        return loss.FocalLoss(sigmoid=True)
    elif loss_name == "GeneralizedDiceLoss":
        return loss.GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "TverskyLoss":
        return loss.TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")