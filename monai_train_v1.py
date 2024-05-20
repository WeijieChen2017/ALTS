import os
import matplotlib.pyplot as plt

from monai.losses import DiceCELoss
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNet

from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
)

import torch

import numpy as np
import nibabel as nib
from matplotlib.colors import ListedColormap

import getpass
import matplotlib

# Setup environment for cache and user
matplotlib.use('Agg')
matplotlib.rcParams['cache.directory'] = '/tmp/matplotlib_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

# Safe user retrieval
try:
    username = getpass.getuser()
except KeyError:
    username = 'default_user'

# Now proceed with your regular imports and script logic
import matplotlib.pyplot as plt

print_config()

data_dir = "data_dir/WORD"
root_dir = "proj_dir/WORD_base"

# creat root directory if not exists
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=2976,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # NormalizeIntensityd(subtrahend=mean_of_dataset, divisor=std_dev_of_dataset),  # mean and std_dev should be precomputed from your dataset
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=2976, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)


import json
import glob

data_json_path = os.path.join(data_dir, "dataset.json")
# load data json
with open(data_json_path, "r") as f:
    data_json = json.load(f)
    
# for key in data_json.keys():
#     print(f"{key}: {data_json[key]}")

training_files = data_json["training"]

# add data directory to each filename
training_files = [{"image": os.path.join(data_dir, d["image"]), "label": os.path.join(data_dir, d["label"])} for d in training_files]
validation_folders = data_json["validation"]
validation_files = []

# search for the file with the same file name in the validation_files
validation_candidates = sorted(glob.glob(os.path.join(data_dir, validation_folders[0], "*.nii.gz")))
for candidate in validation_candidates:
    label_candidate = os.path.join(data_dir, validation_folders[1], os.path.basename(candidate))
    if os.path.exists(label_candidate):
        validation_files.append({"image": candidate, "label": label_candidate})

print(training_files)
print(validation_files)
        

numTraining = len(training_files)
numValidation = len(validation_files)

numClasses = len(data_json["labels"].keys())-1
print("Train:", numTraining, "Validation:", numValidation, "Classes:", numClasses)


# data_dir = "/dataset/"
# split_json = "dataset_0.json"

# datasets = data_dir + split_json
# datalist = load_decathlon_datalist(datasets, True, "training")
# val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = CacheDataset(
    data=training_files,
    transform=train_transforms,
    cache_num=1,
    cache_rate=1.0,
    num_workers=8,
)
val_ds = CacheDataset(
    data=validation_files, 
    transform=val_transforms, 
    cache_num=1,
     cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=numClasses,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

def plot_image(image, label, output, root_dir):

    # colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', 
    #           '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3']
    colors = ['#000000',  # Black for background
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', 
            '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', 
            '#800000', '#aaffc3']
    custom_cmap = ListedColormap(colors)

    epoch = 300
    mean_dice = 0.8
    # plot the middle slice of the image
    idx = image.shape[2] // 2

    plt.figure(figsize=(18, 6), dpi=300)
    plt.subplot(131)
    plt.imshow(np.rot90(image[:, :, idx], 3), cmap="gray")
    plt.axis("off")
    plt.title("image")

    plt.subplot(132)
    plt.imshow(np.rot90(label[:, :, idx], 3), cmap=custom_cmap)
    plt.axis("off")
    plt.title("label")

    plt.subplot(133)
    plt.imshow(np.rot90(output[:, :, idx], 3), cmap=custom_cmap)
    plt.axis("off")
    plt.title("output")

    plt.title(f"Epoch: {epoch}, Mean Dice: {mean_dice}")
    save_path = os.path.join(root_dir, f"epoch_{epoch:.0f}_mean_dice_{mean_dice:.4f}.png")
    plt.savefig(save_path)
    plt.close()

def validation(epoch, epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            val_outputs = model(val_inputs)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038

        image = batch["image"].detach().cpu()
        label = batch["label"].detach().cpu()
        output = val_outputs.detach().cpu()
        plot_image(image, label, output, root_dir)
        mean_dice_val = dice_metric.aggregate().item()
        # print mean dice
        print("Epoch %d Validation Dice: %f" % (epoch, mean_dice_val))
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    # epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(train_loader):
        step += 1
        x, y = (batch["image"].to(device), batch["label"].to(device))
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        # epoch_iterator.set_description(  # noqa: B038
        #     "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        # )
        print("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            # epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(val_loader)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 100
eval_num = 10
post_label = AsDiscrete(to_onehot=numClasses)
post_pred = AsDiscrete(argmax=True, to_onehot=numClasses)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
