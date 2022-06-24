import gc
import numpy as np
import os
import time
import pickle
import glob
from os.path import isfile, join
import matplotlib.pyplot as plt
import torch
import pandas as pd
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_meandice
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    LoadImaged,
    Orientationd,
    CastToTyped,
    RandAffined,
    Rand3DElasticd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from monai.utils import first, set_determinism
from monai.data.utils import partition_dataset
import segmentation_models_pytorch as smp
from monai.metrics.utils import do_metric_reduction, ignore_background
from functions import printandsave

def get_train_val_files(dataset, data_dir, train_rt, val_rt, test_rt, logFile, test_dir, server):
    
    if dataset == 'NSCLC-LLS-2D':
        mask_folder, image_folder = 'mask-slices-numpy', 'image-slices-numpy'
        folder_name = 'LUNG*'
    if dataset == 'NSCLC-LLS-3D':
        mask_folder, image_folder = 'lung-left_lung-right_spinal-cord-numpy', 'image-numpy'
        folder_name = 'LUNG*'
    if dataset == 'OpenKBP-BSMPP-3D' or dataset == 'OH-LLG-3D' or dataset == 'OHC-LLG-3D' or dataset == 'OHC-G-3D':
        mask_folder, image_folder = 'mask-numpy', 'image-numpy'
        folder_name = 'pt*'
    if dataset == 'OpenKBP-BSMPP-2D' or dataset == 'OH-LLG-2D' or dataset == 'OHC-LLG-2D':
        mask_folder, image_folder = 'mask-slices-numpy', 'image-slices-numpy'
        folder_name = 'pt*'
    if dataset == 'PDDCA-BCOOPP-3D':
        mask_folder, image_folder = 'mask-numpy', 'image-numpy'
        folder_name = '0522c*'
    if dataset == 'PDDCA-BCOOPP-2D':
        mask_folder, image_folder = 'mask-slices-numpy', 'image-slices-numpy'
        folder_name = '0522c*'

    patients = sorted(glob.glob(os.path.join(data_dir, folder_name)))
    numOfPatients = len(patients)

    test_num = int(numOfPatients*test_rt)
    val_num = int(numOfPatients*val_rt)
    train_num = int(numOfPatients*train_rt)
    if test_rt+val_rt+train_rt == 1:
        train_num = numOfPatients - test_num - val_num

    printandsave(f'Found {numOfPatients} patients', logFile)

    train_masks = []
    train_images = []
    val_masks = []
    val_images = []
    test_masks = []
    test_images = []

    for patient_num in range(len(patients)):
        patient = patients[patient_num]
        if patient_num<train_num:
            label_dir = glob.glob(os.path.join(patient,mask_folder,'*.npy'))
            ct_dir = glob.glob(os.path.join(patient,image_folder,'*.npy'))
            for i in range(len(label_dir)):
                train_masks.append(label_dir[i])
                train_images.append(ct_dir[i])
        if patient_num>train_num and patient_num<train_num+val_num+1:
            label_dir = glob.glob(os.path.join(patient,mask_folder,'*.npy'))
            ct_dir = glob.glob(os.path.join(patient,image_folder,'*.npy'))
            for i in range(len(label_dir)):
                val_masks.append(label_dir[i])
                val_images.append(ct_dir[i])
        if patient_num>train_num+val_num-1 and patient_num<train_num+val_num+test_num:
            label_dir = glob.glob(os.path.join(patient,mask_folder,'*.npy'))
            ct_dir = glob.glob(os.path.join(patient,image_folder,'*.npy'))
            for i in range(len(label_dir)):
                test_masks.append(label_dir[i])
                test_images.append(ct_dir[i])

    dataset_masks = train_masks + val_masks + test_masks
    dataset_images = train_images + val_images + test_images

    printandsave(f"Total number of patients in the dataset : {numOfPatients}", logFile)
    printandsave(f"Total number of patients we're getting : images: {len(dataset_images)}, labels: {len(dataset_masks)}", logFile)

    train_files = [
        {"image": image_name, "mask": mask_name}
        for image_name, mask_name in zip(train_images, train_masks)
    ]
    val_files = [
        {"image": image_name, "mask": mask_name}
        for image_name, mask_name in zip(val_images, val_masks)
    ]
    test_files = [
        {"image": image_name, "mask": mask_name}
        for image_name, mask_name in zip(test_images, test_masks)
    ]

    printandsave(f"Total number of training data : {len(train_files)}, validation data: {len(val_files)}, test data: {len(test_files)}", logFile)
    
    testLogFile = open(os.path.join(test_dir, f"test_patients.txt"), "w")
    testLogFile.write("Test Files:\n")
    test_patients = []
    for filename in test_files:
        if server:
            ind = 5
        else:
            ind = 6
        pt = filename['image'].split('/')[ind]
        if pt not in test_patients:
            testLogFile.write(f"{pt}\n")
            test_patients.append(pt)
    print(test_patients)
        
    eval_masks = []
    eval_images = []

    for patient in test_patients:
        if dataset.endswith('2D'):
            label_dir = os.path.join(data_dir,patient,mask_folder,'Mask_70.npy')
            ct_dir = os.path.join(data_dir,patient,image_folder,'Image_70.npy')
        if dataset.endswith('3D'):
            label_dir = os.path.join(data_dir,patient,mask_folder,'Mask.npy')
            ct_dir = os.path.join(data_dir,patient,image_folder,'Image.npy')
        eval_masks.append(label_dir)
        eval_images.append(ct_dir)

    eval_files = [
        {"image": image_name, "mask": mask_name}
        for image_name, mask_name in zip(eval_images, eval_masks)
    ]

    #print(next(iter(test_files)),next(iter(eval_files)))
    return train_files, val_files, test_files, eval_files

    
