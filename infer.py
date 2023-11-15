import random
from torchsummary import summary
from torchgeometry.losses import one_hot
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

from dataloader import UNetDataClass, UNetTestDataClass
from model import UNet

parser = argparse.ArgumentParser(description='NeoPolyp Inference')

parser.add_argument('--save_path', type=str, default='/kaggle/working/predicted_masks',
                    help='save path')
parser.add_argument('--pretrained_path', type=str, default="/kaggle/input/unet-checkpoint/unet_model.pth",
                    help='pretrained path')

parser.add_argument('--images_path', type=str, default="/kaggle/input/homework-3-dl/HW3-Dataset/train",
                    help='images path')
parser.add_argument('--masks_path', type=str, default="/kaggle/input/homework-3-dl/HW3-Dataset/train_gt",
                    help='mask path')
parser.add_argument('--test_path', type=str, default='/kaggle/input/homework-3-dl/HW3-Dataset/test/',
                    help='test data path')


args = parser.parse_args()


def weights_init(model):
    if isinstance(model, nn.Linear):
        # Xavier Distribution
        torch.nn.init.xavier_uniform_(model.weight)


def save_model(model, optimizer, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def mask2rgb(mask):
    color_dict = {0: torch.tensor([0, 0, 0]),
                  1: torch.tensor([1, 0, 0]),
                  2: torch.tensor([0, 1, 0])}
    output = torch.zeros(
        (mask.shape[0], mask.shape[1], mask.shape[2], 3)).long()
    for i in range(mask.shape[0]):
        for k in color_dict.keys():
            output[i][mask[i].long() == k] = color_dict[k]
    return output.to(mask.device)


def dice_score(outputs, targets):
    # compute softmax over the classes axis
    output_one_hot = mask2rgb(outputs.argmax(dim=1))

    # create the labels one hot tensor
    target_one_hot = mask2rgb(targets)

    # compute the actual dice score
    dims = (2, 3)
    intersection = torch.sum(output_one_hot * target_one_hot, dims)
    cardinality = torch.sum(output_one_hot + target_one_hot, dims)

    dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
    return dice_score.mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of class in the data set (3: neoplastic, non neoplastic, background)
num_classes = 3

# Number of epoch
epochs = 30

# Hyperparameters for training
learning_rate = 1e-04
batch_size = 8
display_step = 50

# Model path
pretrained_path = args.pretrained_path

# model = UNet()
# # model.apply(weights_init)

# checkpoint = torch.load(pretrained_path)

# # new_state_dict = OrderedDict()
# # for k, v in checkpoint['model'].items():
# #     name = k[7:]  # remove `module.`
# #     new_state_dict[name] = v
# # # load params
# model.load_state_dict(checkpoint['model'])
# model.to(device)
# loss_function = nn.CrossEntropyLoss()

# # Define the optimizer (Adam optimizer)
# optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
# optimizer.load_state_dict(checkpoint['optimizer'])
# new_state_dict = OrderedDict()
# for k, v in checkpoint['model'].items():
#     name = k[7:]  # remove `module.`
#     new_state_dict[name] = v
# # load params
# model.load_state_dict(new_state_dict)

model = UNet()
model = model.to(device)

# Load the checkpoint
checkpoint = torch.load(pretrained_path)

# Print keys for troubleshooting
print("Model State Dictionary Keys:", model.state_dict().keys())
print("Checkpoint Keys:", checkpoint['model'].keys())

# Attempt to load the state dictionary with strict=False
model.load_state_dict(checkpoint['model'], strict=False)

# Move the model to the device
model = model.to(device)

# Define the loss function
loss_function = nn.CrossEntropyLoss()

# Define the optimizer (Adam optimizer)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Load the optimizer state dictionary
optimizer.load_state_dict(checkpoint['optimizer'])


# Test function


def test(dataloader):
    test_loss = 0
    correct = 0
    acc = 0
    dice_acc = 0
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs, 1)
            test_loss += targets.size(0)
            correct += torch.sum(pred == targets).item()
            acc += (outputs.argmax(dim=1) == targets).float().mean()
            dice_acc += dice_score(outputs, targets)
#     print("CHECK: ",100.0 * correct / (test_loss))
    return dice_acc / len(dataloader) * 100, acc/len(dataloader) * 100


# Create submission
transform = Compose([Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                     transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

path = args.test_path
unet_test_dataset = UNetTestDataClass(path, transform)
test_dataloader = DataLoader(unet_test_dataset, batch_size=2, shuffle=True)

model.eval()
if not os.path.isdir("/kaggle/working/predicted_masks"):
    os.mkdir("/kaggle/working/predicted_masks")
for _, (img, path, H, W) in enumerate(test_dataloader):
    a = path
    b = img
    h = H
    w = W

    with torch.no_grad():
        predicted_mask = model(b)
    for i in range(len(a)):
        image_id = a[i].split('/')[-1].split('.')[0]
        filename = image_id + ".png"
        mask2img = Resize((h[i].item(), w[i].item()), interpolation=InterpolationMode.NEAREST)(
            ToPILImage()(F.one_hot(torch.argmax(predicted_mask[i], 0)).permute(2, 0, 1).float()))
        mask2img.save(os.path.join(
            "/kaggle/working/predicted_masks/", filename))


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)


def mask2string(dir):
    # mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:, :, ::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:, :, channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r


MASK_DIR_PATH = '/kaggle/working/predicted_masks'
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(f'output.csv', index=False)
