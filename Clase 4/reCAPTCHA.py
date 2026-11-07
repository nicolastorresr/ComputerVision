#!/usr/bin/env python
# coding: utf-8
# # CLIP reCAPTCHA Solve

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

def load_image(image, resize=None, pil=False):
    if resize is not None:
        image = image.resize((resize, resize))
    if pil:
        return image
    image = np.asarray(image).astype(np.float32) / 255.
    return image

def viz_patches(x, figsize=None, topk=None, t=5, title=None):
    color = (0, 255, 0)
    n = x.shape[0]
    images = []
    for i in range(n):
        im = x[i].permute(1, 2, 0).numpy()
        im = (im * 255.).round().astype(np.uint8)
        if topk is not None:
            if i in topk:
                im[0:t] = color
                im[im.shape[0]-t:] = color
                im[:, 0:t] = color
                im[:, im.shape[1]-t:] = color
        images.append(torch.from_numpy(im).permute(2, 0, 1))
    im = make_grid(images, 3).permute(1, 2, 0).numpy()
    plt.figure(figsize=figsize)
    plt.imshow(im)
    plt.axis('off')
    if title is not None:
        plt.title(title, fontsize=20)
    plt.show()

def patchify(image_path, resolution, patch_size, patch_stride=None, resize=None):
    img_tensor = torch.from_numpy(load_image(image_path, resolution)).permute(2, 0, 1)
    if patch_stride is None:
        patch_stride = patch_size
    patches = img_tensor.unfold(
        1, patch_size, patch_stride).unfold(2, patch_size, patch_stride)
    patches = patches.reshape(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    if resize is not None:
        patches = F.interpolate(
            patches,
            (resize, resize),
            mode='bilinear',
            align_corners=False)
    return patches  # N, 3, patch_size, patch_size

def solve_reCAPTCHA(image, instruction, model, device, topk =  3):
    patch_size = model.visual.input_resolution
    patches = patchify(image, 300, 100, resize=model.visual.input_resolution).to(device)
    text_input = clip.tokenize([instruction]).to(device)

    with torch.no_grad():
        patches_norm = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )(patches)
        patch_embs = model.encode_image(patches_norm).float()
        text_embs = model.encode_text(text_input).float()
        patch_embs = patch_embs / patch_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        sim = patch_embs @ text_embs.t()
        _, topk_idxs = torch.topk(sim.flatten(), topk)
        topk_idxs = topk_idxs.cpu().numpy().tolist()

    viz_patches(
        patches.cpu(),
        figsize=(10, 10),
        topk=topk_idxs,
        t=int(0.05*patch_size),
        title=instruction,
    )








