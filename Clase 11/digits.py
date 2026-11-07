#!/usr/bin/env python
# coding: utf-8

def show_as_image(binary_image, figsize=(20, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(binary_image, cmap='gray')
    plt.xticks([]); plt.yticks([])

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

N_EPOCHS = 25
BATCH_SIZE = 128
LR = 0.005


def causal_mask(width, height, starting_point):
    row_grid, col_grid = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    mask = np.logical_or(
        row_grid < starting_point[0],
        np.logical_and(row_grid == starting_point[0], col_grid <= starting_point[1]))
    return mask

def conv_mask(width, height, include_center=False):
    return 1.0 * causal_mask(width, height, starting_point=(width//2, height//2 + include_center - 1))

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        _, n_channels, width, height = self.weight.size()

        mask = conv_mask(width, height, include_center=mask_type=='B')
        self.register_buffer('mask', torch.from_numpy(mask).float())

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
    
class PixelCNN(nn.Module):
    n_channels = 4
    kernel_size = 7
    padding = 3
    n_pixels_out = 2 # binary 0/1 pixels
    
    def __init__(self):
        super(PixelCNN, self).__init__()
        self.layers = nn.Sequential(
            MaskedConv2d('A', in_channels=1, out_channels=self.n_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False), nn.BatchNorm2d(self.n_channels), nn.ReLU(True),
            MaskedConv2d('B', self.n_channels, self.n_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False), nn.BatchNorm2d(self.n_channels), nn.ReLU(True),
            MaskedConv2d('B', self.n_channels, self.n_channels, kernel_size=self.kernel_size, padding=self.padding, bias=False), nn.BatchNorm2d(self.n_channels), nn.ReLU(True),
            nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_pixels_out, kernel_size=1)
        )
        
    def forward(self, x):
        pixel_logits = self.layers(x)
        return pixel_logits
    
CELL_LENGTH = 4
IMAGE_WIDTH, IMAGE_HEIGHT = 2 * CELL_LENGTH + 5, CELL_LENGTH + 4

def vertical_stroke(rightness, downness):
    """
    Return a 2d numpy array representing an image with a single vertical stroke in it.
    `rightness` and `downness` are values from [0, 1] and define the position of the vertical stroke.
    """
    i = (downness * (CELL_LENGTH + 1)) + 2
    j = rightness * (CELL_LENGTH + 1) + 1
    x = np.zeros(shape=(IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float64)
    x[i + np.arange(CELL_LENGTH), j] = 1.
    return x

def horizontal_stroke(downness):
    """
    Analogue to vertical_stroke, but it returns horizontal strokes.
    `downness` is here a value in [0, 1, 2].
    """
    i = (downness * (CELL_LENGTH + 1)) + 1
    x = np.zeros(shape=(IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float64)
    x[i, 2 + np.arange(CELL_LENGTH)] = 1.
    return x

BASE_STROKES = np.asarray(
    [horizontal_stroke(k) for k in range(3)] + [vertical_stroke(k, l) for k in range(2) for l in range(2)])

DIGITS_STROKES = np.array([[0, 2, 3, 4, 5, 6], [5, 6], [0, 1, 2, 4, 5], 
                           [0, 1, 2, 5, 6], [1, 3, 5, 6], [0, 1, 2, 3, 6], 
                           [0, 1, 2, 3, 4, 6], [0, 5, 6], np.arange(7), [0, 1, 2, 3, 5, 6]], dtype=object)

def random_digits(strokes=BASE_STROKES, digit_as_strokes=DIGITS_STROKES, fixed_label=None):
    label = fixed_label if fixed_label is not None else np.random.choice(len(digit_as_strokes))
    combined_strokes = strokes[digit_as_strokes[label], :, :].sum(axis=0)
    return combined_strokes, label

def batch_images_to_one(batches_images, nrows=10):
    rows_images = np.split(np.squeeze(batches_images), nrows)
    return np.vstack([np.hstack(row_images) for row_images in rows_images])



class LcdDigits(Dataset):

    def __init__(self, n_examples):
        digits, labels = zip(*[random_digits() for _ in range(n_examples)])
        self.digits = np.asarray(digits, dtype=np.float64)
        self.labels = np.asarray(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        digit_with_channel = self.digits[idx][np.newaxis, :, :]
        
        return torch.from_numpy(digit_with_channel).float(), torch.from_numpy(np.array([self.labels[idx]]))


def generate_samples(model, n_samples, starting_point=(0, 0), starting_image=None):

    samples = torch.from_numpy(
        starting_image if starting_image is not None 
        else np.zeros((n_samples * n_samples, 1, IMAGE_WIDTH, IMAGE_HEIGHT))).float()

    model.train(False)

    for i in range(IMAGE_WIDTH):
        for j in range(IMAGE_HEIGHT):
            if i < starting_point[0] or (i == starting_point[0] and j < starting_point[1]):
                continue
            out = model(Variable(samples))
            probs = F.softmax(out[:, :, i, j], dim=1).data
            samples[:, :, i, j] = torch.multinomial(probs, 1).float()
    return samples.numpy()

def all_digits_completion(model,
                    nrows = 1,
                    n_digits = 10, 
                    starting_point = (4, 3), 
                    IMAGE_WIDTH=13, 
                    IMAGE_HEIGHT=8):
    mask = causal_mask(IMAGE_WIDTH, IMAGE_HEIGHT, starting_point)
    starting_images = [random_digits(fixed_label=d)[0] for d in range(n_digits)]
    batch_starting_images = np.expand_dims(np.stack([i * mask for i in starting_images] * nrows), axis=1)
    samples = generate_samples(model, nrows, starting_image=batch_starting_images, 
                               starting_point=starting_point)
    show_as_image(np.hstack([(1 + mask) * i for i in starting_images]), figsize=(20,4))

    show_as_image(
        batch_images_to_one((samples * (1 + mask)), nrows),
        figsize=(20,4*nrows))

def digit_completion(model,
                    digit,
                    n_digits = 10, 
                    starting_point = (4, 3), 
                    IMAGE_WIDTH=13, 
                    IMAGE_HEIGHT=8):
    mask = causal_mask(IMAGE_WIDTH, IMAGE_HEIGHT, starting_point)
    if n_digits < 2:
        n_digits = 2
    
    starting_images = [random_digits(fixed_label=digit)[0] for _ in range(n_digits)]
   
    batch_starting_images = np.expand_dims(np.stack([i * mask for i in starting_images]), axis=1)
   
    samples = generate_samples(model, n_digits, starting_image=batch_starting_images, 
                               starting_point=starting_point)
    
    show_as_image(starting_images[0]+mask, figsize=(4,4))

    show_as_image(
        batch_images_to_one((samples * (1 + mask)), 1),
        figsize=(20,4*1))