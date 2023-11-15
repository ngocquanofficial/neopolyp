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
parser.add_argument('--model', type=str, default='model.pth',
                    help='model path')
parser.add_argument('--data_path', type=str, default='data',
                    help='data path')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--save_path', type=str, default='/kaggle/working/predicted_masks',
                    help='save path')
parser.add_argument('--csv_path', type=str, default='/kaggle/working/',
                    help='csv path')
parser.add_argument('--checkpoint_path', type=str, default='/kaggle/working/unet_model.pth',
                    help='checkpoint path')
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
checkpoint_path = args.checkpoint_path
pretrained_path = args.pretrained_path
# Initialize lists to keep track of loss and accuracy
loss_epoch_array = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []
dice_score_train = []
dice_score_val = []

train_size = 0.8
valid_size = 0.2
images_path = args.images_path
masks_path = args.masks_path
images_list = os.listdir(images_path)
masks_list = os.listdir(masks_path)
training_index = random.sample(
    range(len(images_list)), int(len(images_list) * train_size))
validation_index = [i for i in range(
    len(images_list)) if i not in training_index]
# training_images_list = [images_list[i] for i in training_index]
# validation_images_list = [images_list[i] for i in validation_index]


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

unet_training_dataset = UNetDataClass(
    images_path, masks_path, training_transform, training_index)
unet_validation_dataset = UNetDataClass(
    images_path, masks_path, validation_transform, validation_index)

train_dataloader = DataLoader(
    unet_training_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(
    unet_validation_dataset, batch_size=batch_size, shuffle=True)

model = UNet()
model.apply(weights_init)
model = nn.DataParallel(model)

checkpoint = torch.load(pretrained_path)

new_state_dict = OrderedDict()
for k, v in checkpoint['model'].items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
model = nn.DataParallel(model)
model.to(device)
loss_function = nn.CrossEntropyLoss()

# Define the optimizer (Adam optimizer)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# optimizer.load_state_dict(checkpoint['optimizer'])

# Learning rate scheduler
learing_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.6)


def train(train_dataloader, valid_dataloader, learing_rate_scheduler, epoch, display_step):
    print(
        f"Start epoch #{epoch+1}, learning rate for this epoch: {learing_rate_scheduler.get_last_lr()}")
    start_time = time.time()
    train_loss_epoch = 0
    test_loss_epoch = 0
    last_loss = 999999999
    model.train()
    for i, (data, targets) in enumerate(train_dataloader):

        # Load data into GPU
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        # Backpropagation, compute gradients
        loss = loss_function(outputs, targets.long())
        loss.backward()

        # Apply gradients
        optimizer.step()

        # Save loss
        train_loss_epoch += loss.item()
        if (i+1) % display_step == 0:
            #             accuracy = float(test(test_loader))
            print('Train Epoch: {} [{}/{} ({}%)]\tLoss: {:.4f}'.format(
                epoch + 1, (i+1) * len(data), len(train_dataloader.dataset), 100 *
                (i+1) * len(data) / len(train_dataloader.dataset),
                loss.item()))

    print(
        f"Done epoch #{epoch+1}, time for this epoch: {time.time()-start_time}s")
    train_loss_epoch /= (i + 1)

    # Evaluate the validation set
    model.eval()
    with torch.no_grad():
        for data, target in valid_dataloader:
            data, target = data.to(device), target.to(device)
            test_output = model(data)
            test_loss = loss_function(test_output, target)
            test_loss_epoch += test_loss.item()

    test_loss_epoch /= (i+1)

    return train_loss_epoch, test_loss_epoch


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


wandb.login(
    # set the wandb project where this run will be logged
    #     project= "PolypSegment",
    key="9e7c7aead65864c7d2a322f932d2e9a6e2d631d9",
)
wandb.init(
    project="PolypSegment"
)
# Training loop
train_loss_array = []
test_loss_array = []
last_loss = 9999999999999
for epoch in range(epochs):
    print(f"Start at epoch {epoch}")
    train_loss_epoch = 0
    test_loss_epoch = 0
    (train_loss_epoch, test_loss_epoch) = train(train_dataloader,
                                                valid_dataloader,
                                                learing_rate_scheduler, epoch, display_step)

    if test_loss_epoch < last_loss:
        save_model(model, optimizer, checkpoint_path)
        last_loss = test_loss_epoch

    learing_rate_scheduler.step()
    train_loss_array.append(train_loss_epoch)
    test_loss_array.append(test_loss_epoch)
    wandb.log({"Train loss": train_loss_epoch, "Valid loss": test_loss_epoch})
    train_accuracy.append(test(train_dataloader)[1])
    valid_accuracy.append(test(valid_dataloader)[1])

    dice_score_train.append(test(train_dataloader)[0])
    dice_score_val.append(test(train_dataloader)[0])
    print("Epoch {}: loss: {:.4f}, train accuracy: {:.4f}, valid accuracy:{:.4f}".format(epoch + 1,
                                                                                         train_loss_array[-1], train_accuracy[-1], valid_accuracy[-1]))
    print(
        f"Dice score train: {dice_score_train[-1]}, val: {dice_score_val[-1]}")


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


# change this to the path to your output mask folder
MASK_DIR_PATH = '/kaggle/working/predicted_masks'
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(f'output.csv', index=False)
