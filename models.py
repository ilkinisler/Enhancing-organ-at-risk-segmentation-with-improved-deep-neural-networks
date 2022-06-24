
import segmentation_models_pytorch as smp
from monai.networks.layers import Norm
from monai.networks.nets import UNet, UNETR
from dilatedUNet import dilatedUNet


def unet2D(encoder, numberofclasses):
    model = smp.Unet(
        encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=numberofclasses,                      # model output channels (number of classes in your dataset)
    )
    return model

def unetplusplus(encoder, numberofclasses):
    model = smp.UnetPlusPlus(
        encoder_name=encoder, 
        encoder_weights='imagenet', 
        in_channels=1, 
        classes=numberofclasses, 
    )
    return model

def monaiunet(dimension, numberofclasses):
    model = UNet(
        dimensions=dimension,
        in_channels=1,
        out_channels=numberofclasses,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return model

def dilatedmonaiunet(dimension, numberofclasses, dilation):
    model = dilatedUNet(
        dimensions=dimension,
        in_channels=1,
        out_channels=numberofclasses,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
        num_dilation=dilation
    )
    return model

def unetr(numberofclasses, img_size):
    model = UNETR(
        in_channels=1,
        out_channels=numberofclasses,
        img_size=img_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )
    return model