import torch
import random
import numpy as np
import timm
from timm import create_model

from models.SSDCA import SiameseSwinSCross_dual
from models.SiameseSwin_Small import SiameseSwinS


def get_model(model_name, layer, num_classes):
    if model_name == "Swin_S":
        model = create_model("swin_small_patch4_window7_224", pretrained=True, num_classes=num_classes)
    elif model_name == "SSDCA":
        model = SiameseSwinSCross_dual()
    elif model_name == "SSFC":
        model = SiameseSwinS()
    else:
        raise ValueError("Invalid model name. Choose from Resnet50, WRN50, ViT, Swin, custom_Swin")
    return model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


