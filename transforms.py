
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    RandAdjustContrastd,
    AdjustContrastd,
    LoadImaged,
    Orientationd,
    CastToTyped,
    RandAffined,
    Rand3DElasticd,
    Rand2DElasticd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    DataStats,
    RandGaussianNoised,
)
import numpy as np


def getTrainTransform(dimension, aug_roi):
    if dimension == 3:
        train_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            AddChanneld(keys=["image", "mask"]),
            CastToTyped(keys=['image',"mask"],dtype='float32'),
            RandAdjustContrastd(keys=["image"],gamma=6),
            NormalizeIntensityd(keys=['image']),
            RandCropByPosNegLabeld(
                keys=["image", "mask"],
                label_key="mask",
                spatial_size=aug_roi,
                pos=1,
                neg=1,
                num_samples=16,
                image_key="image",
                image_threshold=0,
            ),
            RandAffined(
                keys=["image", "mask"],
                mode=("bilinear", "nearest"),
                prob=0.1,
                spatial_size=aug_roi,
                translate_range=(40, 40, 2),
                rotate_range=(np.pi / 6, np.pi / 18, np.pi /18),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="border",
            ),
            Rand3DElasticd(
                keys=["image", "mask"],
                mode=("bilinear", "nearest"),
                prob=0.1,
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                spatial_size=aug_roi,
                translate_range=(50, 50, 2),
                rotate_range=(np.pi, np.pi / 36, np.pi / 36),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="border",
            ),
            RandGaussianNoised(
                keys=["image", "mask"], 
                prob=0.1,
                mean=0.0,
                std=0.1,
                allow_missing_keys=False,
            ),
            ToTensord(keys=["image", "mask"]),
        ])
    if dimension == 2:
        train_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            AddChanneld(keys=["image", "mask"]),
            CastToTyped(keys=['image',"mask"],dtype='float32'),
            RandAdjustContrastd(keys=["image"],gamma=6),
            NormalizeIntensityd(keys=['image']),
            RandCropByPosNegLabeld(
                keys=["image", "mask"],
                label_key="mask",
                spatial_size=aug_roi,
                pos=1,
                neg=1,
                num_samples=16,
                image_key="image",
                image_threshold=0,
            ),
            RandAffined(
                keys=["image", "mask"],
                mode=("bilinear", "nearest"),
                prob=0.1,
                spatial_size=aug_roi,
                translate_range=(40, 40),
                rotate_range=(np.pi / 18, np.pi /18),
                scale_range=(0.15, 0.15),
                padding_mode="border",
            ),
            Rand2DElasticd(
                keys=["image", "mask"], 
                prob=0.1,
                spacing=(30, 30),
                magnitude_range=(5, 6),
                rotate_range=(np.pi / 4,),
                scale_range=(0.2, 0.2),
                translate_range=(100, 100),
                padding_mode="zeros",
            ),
            RandGaussianNoised(
                keys=["image", "mask"], 
                prob=0.1,
                mean=0.0,
                std=0.1,
                allow_missing_keys=False,
            ),
            ToTensord(keys=["image", "mask"]),
        ])

    return train_transforms

def getValTransform():
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            AddChanneld(keys=["image", "mask"]),
            CastToTyped(keys=['image',"mask"],dtype='float32'),
            NormalizeIntensityd(keys=['image']),
            #RandAdjustContrastd(keys=["image"]),
            ToTensord(keys=["image", "mask"]),
        ]
    )
    return val_transforms