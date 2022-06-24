# -*- coding: utf-8 -*-

caption = 'unettrial1'
trial = 'ur1'
dataset = 'OpenKBP-BSMPP-3D'
architecture = "unetr"
encoder_name= "efficientnet-b4" #only for unet2D eg. #resnet34 #efficientnet-b4
epoch_num = 5
diceCE = True
wdice, wce = 0.3, 0.7
server = True
dilation = 3 #only for dilated models
cyclictype = 'exp_range' #triangular,triangular2,exp_range
#train_rt, val_rt, test_rt = 0.30, 0.10, 0.10
train_rt, val_rt, test_rt = 0.70, 0.15, 0.15

import time
from datetime import datetime
import gc
import warnings
import numpy as np
import os
import pickle
import glob
import torch
import torch.nn as nn
import pandas as pd
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric, compute_meandice, compute_hausdorff_distance, HausdorffDistanceMetric
from monai.utils import first, set_determinism
import segmentation_models_pytorch as smp
from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
    Activations,
)
import functions
import models
import loaddata
import transforms
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

print_config()
torch.cuda.empty_cache()

datasets = ['NSCLC-LLS-2D','NSCLC-LLS-3D','OpenKBP-BSMPP-2D','OpenKBP-BSMPP-3D','PDDCA-BCOOPP-3D','PDDCA-BCOOPP-2D','OH-LLG-3D','OH-LLG-2D','OHC-LLG-3D','OHC-LLG-2D','OHC-G-3D']
architectures = ['unet2D','monaiunet2D','monaiunet3D','unet++', 'dilatedmonaiunet2D', 'dynunet2D', 'dynunet3D','unetr']

lossfunc = functions.get_loss_func(diceCE)
label_names, numberofclasses = functions.get_classes(dataset) 
dimension,batchsize = functions.get_dimension_and_bs(dataset)
dice_roi,aug_roi = functions.get_rois(dataset, architecture)

datasetID = datasets.index(dataset)
architectureID = architectures.index(architecture)
trial_name = f"{trial}_{architectureID}_{datasetID}"

if server:
    root_dir = '/home/ilkin/FDOH'
    data_dir = os.path.join("/home/ilkin/data", dataset)
else:
    root_dir = '/home/ilkinisler/Documents/FDOH'
    data_dir = os.path.join("/home/ilkinisler/Documents/data", dataset)
model_dir = os.path.join(root_dir, "models")
test_dir = os.path.join(root_dir, "tests", architecture, dataset, trial_name)

loss_metric_path = os.path.join(test_dir, f"{trial}_loss_dice.png")
class_dice_path = os.path.join(test_dir, f"{trial}_class_dice.png")
class_hd_path = os.path.join(test_dir, f"{trial}_class_hd.png")

best_metric_model_path = os.path.join(model_dir, f"{trial_name}_best_model.pth")

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(os.path.join(root_dir, "tests", architecture, dataset)):
    os.mkdir(os.path.join(root_dir, "tests", architecture, dataset))
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
logFile = open(os.path.join(test_dir, f"{trial_name}_train_out.txt"), "w")
logFile.write(f"Date & time: {dt_string} \n"
            f"Caption: {caption} \n"
            f"Trial number: {trial} \n"
            f"Architecture: {architecture} \n"
            f"Encoder: {encoder_name} \n"
            f"Dataset: {dataset} \n"
            f"Trial code: {trial_name} \n"
            f"Batch size: {batchsize}\n"
            f"Loss function: {lossfunc} ({wdice}-{wce})\n"
            f"CcylicLR: {cyclictype}\n")

train_files, val_files, test_files, eval_files = loaddata.get_train_val_files(dataset, data_dir, train_rt, val_rt, test_rt, logFile, test_dir, server)

set_determinism(seed=0)

train_transforms = transforms.getTrainTransform(dimension,aug_roi)
val_transforms = transforms.getValTransform()

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.1, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.1, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

test_transforms = transforms.getValTransform()

test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=0.1, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

eval_ds = CacheDataset(data=eval_files, transform=test_transforms, cache_rate=0.1, num_workers=4)
eval_loader = DataLoader(eval_ds, batch_size=1, num_workers=4)

device = torch.device("cuda")

if architecture=='unet2D':
    model = models.unet2D(encoder_name, numberofclasses+1)
if architecture=='unet++':
    model =  models.unetplusplus(encoder_name, numberofclasses+1)
if architecture in ['monaiunet2D','monaiunet3D']:
    model =  models.monaiunet(dimension, numberofclasses+1)
if architecture in ['dynunet2D','dynunet3D']:
    model =  models.dynunet(dimension, numberofclasses+1)
if architecture in ['dilatedmonaiunet2D']:
    model =  models.dilatedmonaiunet(dimension, numberofclasses+1, dilation)
if architecture=='unetr':
    model = models.unetr(numberofclasses+1, aug_roi)

if server:
    model = nn.DataParallel(model)

model.to(device)

if diceCE:
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.4, lambda_ce=0.6)
else:
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
scheduler = CyclicLR(optimizer, mode=cyclictype, max_lr=0.003, base_lr=0.0005, cycle_momentum=False)
warnings.filterwarnings('ignore')

post_pred = AsDiscrete(argmax=True, to_onehot=numberofclasses+1)
post_label = AsDiscrete(to_onehot=numberofclasses+1)

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_cb = [[] for i in range(numberofclasses)]

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["mask"].cuda())
            val_outputs = sliding_window_inference(val_inputs, aug_roi, 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()

    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["mask"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            
            statedict = model.state_dict()
            
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(statedict, best_metric_model_path) 
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 20000
eval_num = 400
post_label = AsDiscrete(to_onehot=numberofclasses+1)
post_pred = AsDiscrete(argmax=True, to_onehot=numberofclasses+1)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
t0 = time.time()
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
t1 = time.time()

logFile.write(f"Max iterations: {max_iterations}      Eval number: {eval_num}\n")

model_results = (f"\nTrain completed: \nBest_metric: {dice_val_best:.4f} "
    f"at iteration: {global_step_best}\ntime: {((t1-t0)/60):.4f}")

print(model_results)
logFile.write(f"{model_results}\n")

model.load_state_dict(torch.load(best_metric_model_path))

model.eval()

functions.avg_loss_metric_graph(epoch_loss_values, metric_values, eval_num, loss_metric_path)

saved_metrics = {'max_iterations':max_iterations,'metric_values':metric_values, 'epoch_loss_values':epoch_loss_values}
object = saved_metrics 
filehandler = open(os.path.join(root_dir, "tests", f"{architecture}/{dataset}/{trial}_{architectureID}_{datasetID}/{trial}_saved_metrics_{epoch_num}_epochs.obj"), 'wb') 
pickle.dump(object, filehandler)

meandice = np.zeros([numberofclasses+1])
k = 0
for i, test_data in enumerate(test_loader):
    test_inputs, test_labels = (
    test_data["image"].to(device),
    test_data["mask"].to(device),
    )
    meandice += functions.calculate_dice(test_inputs,test_labels,aug_roi,model)
    k += 1
print(k, "test images")
meandice = meandice/k
test_results = (f"\nTest completed: \nBest metrics: {np.array2string(meandice)} \nBest mean metric: {(np.nanmean(meandice)):.4f}")

print(test_results)
logFile.write(f"{test_results}\n")

functions.save_evaluation_results(dataset, test_loader, model, test_dir, trial, device, eval_files, aug_roi)

logFile.close() 
