import io

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

import numpy as np
import pandas as pd
import seaborn as sn
import torch
from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

from torchvision import models
import json


model = models.densenet121(pretrained=True)
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((255)),
                                        #transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


imagenet_class_index = json.load(open('imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

if __name__ == '__main__':
    with open("img.jpeg", 'rb') as f:
        image_bytes = f.read()
        print(get_prediction(image_bytes=image_bytes))
        