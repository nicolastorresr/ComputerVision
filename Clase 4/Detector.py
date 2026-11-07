#!/usr/bin/env python
# coding: utf-8
# # CLIP Zero-shot Detector

from collections import OrderedDict
import clip
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import selectivesearch
import torch
from PIL import Image

def load_image(img_path, resize=None, pil=False):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    if pil:
        return image
    image = np.asarray(image).astype(np.float32) / 255. 
    return image

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def vis_detections(im, dets, thresh=0.5, caption=None, show_all=False):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    top_idx = dets[:, -1].argmax()

    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        if i == top_idx:
            ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='green', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:.3f}'.format(score),
                    bbox=dict(facecolor='green', alpha=1),
                    fontsize=14, color='white')
        else:
            if show_all:
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                bbox[2] - bbox[0],
                                bbox[3] - bbox[1], fill=False,
                                edgecolor='blue', linewidth=3.5)
                    )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:.3f}'.format(score),
                        bbox=dict(facecolor='blue', alpha=1),
                        fontsize=14, color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if caption is not None:
        plt.title(caption, fontsize=20)
    plt.show()

def plot_selective_search(img,
           resize = None, 
           topk = 50, 
           scale =  200, 
           sigma =  0.8, 
           min_size = 50):
    if resize is not None:
        assert isinstance(resize, int), "resize should be an integer."
    oh, ow = img.shape[:2]
    # Selective search.
    img_search = img.copy()
    img_lbl, regions = selectivesearch.selective_search(
        img_search, scale=scale, sigma=sigma, min_size=min_size)
    candidates = OrderedDict()
    for i, r in enumerate(regions):
        if r['rect'] in candidates:
            continue
        if r['size'] < 1000:
            continue
        x, y, w, h = r['rect']
        if w / h > 1.5 or h / w > 1.5:
            continue
        if resize is not None:
            sx = (ow / resize)
            sy = (oh / resize)
            x_, y_, w_, h_ = r['rect']
            x = np.clip(x_ * sx, 0, ow).astype(int)
            y = np.clip(y_ * sy, 0, oh).astype(int)
            w = np.clip(w_ * sx, 0, ow).astype(int)
            h = np.clip(h_ * sy, 0, oh).astype(int)
            r['rect'] = (x, y, w, h)
        candidates[i] = r['rect']
    candidates = list(candidates.values())
    candidates = candidates[:topk]
    # Display topk bounding boxes.
    _, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    ax.imshow(img)
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()
    return candidates

def detect(img, 
           candidates,
           image_caption, 
           model, 
           preprocess,
           device, 
           show_all = False,
           nms_threshold =  0.1,
           viz_threshold = 0.2):
    batch = []
    for box in candidates:
        x, y, w, h = box
        patch = (img[y:y+h, x:x+w] * 255.).astype(np.uint8)
        patch = Image.fromarray(patch).convert("RGB")
        patch = preprocess(patch).unsqueeze(0)
        batch.append(patch)
    batch = torch.cat(batch, dim=0).to(device)

    text_input = clip.tokenize(image_caption).to(device)

    with torch.no_grad():
        patch_embs = model.encode_image(batch).float()
        text_embs = model.encode_text(text_input).float()
        patch_embs = patch_embs / patch_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        sim = patch_embs @ text_embs.t()
    scores = sim.cpu().numpy().T[0]

    dets = []
    for box, score in zip(candidates, scores):
        x, y, w, h = box
        x1 = x
        x2 = x1 + w
        y1 = y
        y2 = y1 + h
        dets.append([x1, y1, x2, y2, score])
    dets = np.asarray(dets)
    keep = nms(dets, nms_threshold)
    dets_keep = dets[keep]

    vis_detections(img, 
                   dets_keep, 
                   thresh=viz_threshold, 
                   caption="Detecting "+image_caption,
                   show_all=show_all)
