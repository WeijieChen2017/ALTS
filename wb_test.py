import wandb
import random
import time

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="wb-test_proj_01",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "architecture": "U-Net",
    "dataset": "WORD",
    "epochs": 100,
    }
)

# simulate training
epochs = 100
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # pause for 1 s
    time.sleep(1)
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()