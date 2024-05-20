# load config file
import json
config = json.load(open('config/20240509.json'))
root_dir = "proj_dir/"+config["proj_dir"]+"/"
from utils import str_to_tuple
from utils import str_to_boolean

# create a model
from model import UNet_Theseus
from monai.networks.nets import UNet
model = UNet(
    spatial_dims=config["model_params"]["spatial_dims"],
    in_channels=config["model_params"]["in_channels"],
    out_channels=config["model_params"]["out_channels"],
    channels=str_to_tuple(config["model_params"]["channels"]),
    strides=str_to_tuple(config["model_params"]["strides"]),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=2,
    act='PRELU',
    norm='INSTANCE',
    dropout=0.0,
    bias=True,
    adn_ordering='NDA'
)
# model = UNet_Theseus(
#     spatial_dims = config["model_params"]["spatial_dims"],
#     in_channels = config["model_params"]["in_channels"],
#     out_channels = config["model_params"]["out_channels"],
#     channels = str_to_tuple(config["model_params"]["channels"]),
#     strides = str_to_tuple(config["model_params"]["strides"]),
#     kernel_size = config["model_params"]["kernel_size"],
#     up_kernel_size = config["model_params"]["up_kernel_size"],
#     num_res_units = config["model_params"]["num_res_units"],
#     act = config["model_params"]["act"],
#     norm = config["model_params"]["norm"],
#     dropout = config["model_params"]["dropout"],
#     bias = str_to_boolean(config["model_params"]["bias"]),
#     adn_ordering = config["model_params"]["adn_ordering"],
#     dimensions = config["model_params"]["dimensions"],
#     alter_block = config["model_params"]["alter_block"],
# )

# create a loss
loss_name = config["train_params"]["loss"]
from utils import get_loss
loss = get_loss(loss_name)

# create a optimizer
optimizer_name = config["train_params"]["optimizer"]["name"]
lr = config["train_params"]["optimizer"]["lr"]
weight_decay = config["train_params"]["optimizer"]["weight_decay"]
from utils import get_optimizer
optimizer = get_optimizer(optimizer_name, model, lr, weight_decay)

# create a scheduler
scheduler_name = config["train_params"]["scheduler"]["name"]
scheduler_T_max = config["train_params"]["scheduler"]["T_max"]
scheduler_eta_min = config["train_params"]["scheduler"]["eta_min"]
from utils import get_scheduler
scheduler = get_scheduler(scheduler_name, optimizer, scheduler_T_max, scheduler_eta_min)

# create a data loader
import os
import glob
data_folder = config["train_params"]["data_folder"]
data_list = sorted(glob.glob(os.path.join(data_folder, "*/")))
training_ratio = config["train_params"]["training_ratio"]
validation_ratio = config["train_params"]["validation_ratio"]
testing_ratio = config["train_params"]["testing_ratio"]

# shuffle and divide the data list
import random
random.shuffle(data_list)
train_list = data_list[:int(len(data_list)*training_ratio)]
valid_list = data_list[int(len(data_list)*training_ratio):int(len(data_list)*(training_ratio+validation_ratio))]
test_list = data_list[int(len(data_list)*(training_ratio+validation_ratio)):]
print("train_list:", train_list)
print("valid_list:", valid_list)
print("test_list:", test_list)
# save the division into a txt file
with open(root_dir+"data_list.txt", "w") as f:
    f.write("train_list: "+str(train_list)+"\n")
    f.write("valid_list: "+str(valid_list)+"\n")
    f.write("test_list: "+str(test_list)+"\n")

# start the training
from utils import get_data
n_epoch = config["train_params"]["n_epoch"]
batch_size = config["train_params"]["batch_size"]
res_xyz = str_to_tuple(config["train_params"]["res_xyz"])
res_xyz = tuple([int(i) for i in res_xyz[1:-1].split(",")])
out_channels = config["model_params"]["out_channels"]
for idx_epoch in range(n_epoch):
    # start the training
    model.train()
    for idx_batch in range(0, len(train_list), batch_size):
        # load the data
        dir_list = train_list[idx_batch:idx_batch+batch_size]
        # do the training
        batch_x, batch_y = get_data(dir_list, batch_size, res_xyz, out_channels)
            
