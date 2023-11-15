import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import time
import imageio
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, PILToTensor, ToPILImage, Compose, InterpolationMode
from collections import OrderedDict
import wandb
import argparse

import torchvision.transforms as transforms


# Define probabilities for each transform
p_horizontal_flip = 0.5
p_vertical_flip = 0.3
p_rotation = 0.3
p_random_crop = 0.1
p_gaussian_blur = 0.2  # Adjust based on preference
p_color_jitter = 0.3

training_transform = transforms.Compose([
    transforms.RandomApply(
        [transforms.RandomHorizontalFlip()], p=p_horizontal_flip),
    transforms.RandomApply(
        [transforms.RandomVerticalFlip()], p=p_vertical_flip),
    transforms.RandomApply(
        [transforms.RandomRotation(degrees=15)], p=p_rotation),
    transforms.RandomApply(
        [transforms.RandomResizedCrop(256)], p=p_random_crop),
    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=3)], p=p_gaussian_blur),
    transforms.RandomApply([transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=p_color_jitter),

    transforms.Resize(
        (256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])
validation_transform = Compose([transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
                                transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

transform = Compose([Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                     transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


class UNetDataClass(Dataset):
    def __init__(self, images_path, masks_path, transform, image_index):
        super(UNetDataClass, self).__init__()

        full_images_list = os.listdir(images_path)
        full_masks_list = os.listdir(masks_path)
        images_list = [full_images_list[i] for i in image_index]
        masks_list = [full_masks_list[i] for i in image_index]
#         print(images_list)

        # Full path to images
        images_list = [os.path.join(images_path, image_name)
                       for image_name in images_list]
        masks_list = [os.path.join(masks_path, mask_name)
                      for mask_name in masks_list]

        self.images_list = images_list
        self.masks_list = masks_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.images_list[index]
        mask_path = self.masks_list[index]

        # Open image and mask
        data = Image.open(img_path)
        label = Image.open(mask_path)

        # Normalize
        data = self.transform(data)
        label = self.transform(label)

        label = torch.where(label > 0.65, 1.0, 0.0)

        label[2, :, :] = 0.0001
        label = torch.argmax(label, 0).type(torch.int64)
        return data, label

    def __len__(self):
        return len(self.images_list)


class UNetTestDataClass(Dataset):
    def __init__(self, images_path, transform):
        super(UNetTestDataClass, self).__init__()

        images_list = os.listdir(images_path)
        images_list = [images_path+i for i in images_list]

        self.images_list = images_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.images_list[index]
        data = Image.open(img_path)
        h = data.size[1]
        w = data.size[0]
        data = self.transform(data)
        return data, img_path, h, w

    def __len__(self):
        return len(self.images_list)
