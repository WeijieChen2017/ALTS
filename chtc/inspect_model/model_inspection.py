# here we have V1/
# total 477M
# -rw-r--r-- 1 wchen376 wchen376 476M Mar 31  2022 model_final_checkpoint.model
# -rw-r--r-- 1 wchen376 wchen376  64K Mar 31  2022 model_final_checkpoint.model.pkl
# -rw-r--r-- 1 wchen376 wchen376  63K Mar 31  2022 plans.pkl
# -rw-r--r-- 1 wchen376 wchen376 337K Mar 31  2022 progress.png
# -rw-r--r-- 1 wchen376 wchen376 427K Mar 31  2022 training_log_2021_9_18_12_39_13.txt
# and we want to load the model_final_checkpoint.model, model_final_checkpoint.model.pkl, and plans.pkl to inspect the model
# output the model structure and the training plan into model_structure.txt and training_plan.txt

import torch
import pickle
import os
import sys
import numpy as np

model_path = {
    "model": "model_final_checkpoint.model",
    "model_pkl": "model_final_checkpoint.model.pkl",
    "plans": "plans.pkl"
}

# load the model
model = torch.load(model_path["model"])
# write all keys in model into model_structure.txt
with open("model_structure.txt", "w") as f:
    for key in model.keys():
        f.write(str(key)+"\n")

# load the model.pkl
with open(model_path["model_pkl"], "rb") as f:
    model_pkl = pickle.load(f)
# write model_pkl into model_pkl.txt
with open("model_pkl.txt", "w") as f:
    f.write(str(model_pkl))

# load the plans
with open(model_path["plans"], "rb") as f:
    plans = pickle.load(f)
# write plans into plans.txt
with open("plans.txt", "w") as f:
    f.write(str(plans))