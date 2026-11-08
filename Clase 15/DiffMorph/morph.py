#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

@tf.function
def warp(origins, targets, preds_org, preds_trg, add_scale, mult_scale, im_sz, warp_scale, add_first):
    if add_first:
        res_targets = tfa.image.dense_image_warp((origins + preds_org[:,:,:,3:6] * 2 * add_scale) * tf.maximum(0.1, 1 + preds_org[:,:,:,0:3] * mult_scale) , preds_org[:,:,:,6:8] * im_sz * warp_scale )
        res_origins = tfa.image.dense_image_warp((targets + preds_trg[:,:,:,3:6] * 2 * add_scale) * tf.maximum(0.1, 1 + preds_trg[:,:,:,0:3] * mult_scale) , preds_trg[:,:,:,6:8] * im_sz * warp_scale )
    else:
        res_targets = tfa.image.dense_image_warp(origins * tf.maximum(0.1, 1 + preds_org[:,:,:,0:3] * mult_scale) + preds_org[:,:,:,3:6] * 2 * add_scale, preds_org[:,:,:,6:8] * im_sz * warp_scale )
        res_origins = tfa.image.dense_image_warp(targets * tf.maximum(0.1, 1 + preds_trg[:,:,:,0:3] * mult_scale) + preds_trg[:,:,:,3:6] * 2 * add_scale, preds_trg[:,:,:,6:8] * im_sz * warp_scale )

    return res_targets, res_origins

def create_grid(scale):
    grid = np.mgrid[0:scale,0:scale] / (scale - 1) * 2 -1
    grid = np.swapaxes(grid, 0, 2)
    grid = np.expand_dims(grid, axis=0)
    return grid

def produce_maps(source, 
                target, 
                train_epochs = 1000, 
                warp_scale = 0.05,
                add_scale = 0.4, 
                mult_scale = 0.4,
                add_first = False,
                im_sz = 1024, 
                mp_sz = 96,
                output_name = "saved/preds.npy"):
    
    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(64, (5, 5))
            self.act1 = tf.keras.layers.LeakyReLU(alpha=0.2)
            self.conv2 = tf.keras.layers.Conv2D(64, (5, 5))
            self.act2 = tf.keras.layers.LeakyReLU(alpha=0.2)
            self.convo = tf.keras.layers.Conv2D((3 + 3 + 2) * 2, (5, 5))

        def call(self, maps):
            x = tf.image.resize(maps, [mp_sz, mp_sz])
            x = self.conv1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.act2(x)
            x = self.convo(x)
            return x

    if not source:
        print("No source file provided!")
        exit()

    if not target:
        print("No target file provided!")
        exit()


    dom_a = cv2.imread(source, cv2.IMREAD_COLOR)
    dom_b = cv2.imread(target, cv2.IMREAD_COLOR)

    # Checks if input and destination image are of the same dimensions.
    if (dom_a.shape[1] != dom_b.shape[1] or dom_a.shape[0] != dom_b.shape[0]):
        print("Input Image is not the same dimensions as Destination Image.")
        sys.exit()

    dom_a = cv2.cvtColor(dom_a, cv2.COLOR_BGR2RGB)
    dom_a = cv2.resize(dom_a, (im_sz, im_sz), interpolation = cv2.INTER_AREA)
    dom_a = dom_a / 127.5 - 1

    dom_b = cv2.cvtColor(dom_b, cv2.COLOR_BGR2RGB)
    dom_b = cv2.resize(dom_b, (im_sz, im_sz), interpolation = cv2.INTER_AREA)
    dom_b = dom_b / 127.5 - 1

    origins = dom_a.reshape(1, im_sz, im_sz, 3).astype(np.float32)
    targets = dom_b.reshape(1, im_sz, im_sz, 3).astype(np.float32)

    model = MyModel()

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(maps, 
                   origins, 
                   targets, 
                   warp_scale = 0.05, 
                   add_scale = 0.4, 
                   mult_scale = 0.4, 
                   add_first = False, 
                   im_sz = 1024, 
                   ):
      
      with tf.GradientTape() as tape:
        preds = model(maps)
        preds = tf.image.resize(preds, [im_sz, im_sz])

        res_targets_, res_origins_ = warp(origins, targets, preds[...,:8], preds[...,8:], add_scale, mult_scale, im_sz, warp_scale, add_first)

        res_map = tfa.image.dense_image_warp(maps, preds[:,:,:,6:8] * im_sz * warp_scale ) #warp maps consistency checker
        res_map = tfa.image.dense_image_warp(res_map, preds[:,:,:,14:16] * im_sz * warp_scale )

        loss = loss_object(maps, res_map) * 1 + loss_object(res_targets_, targets) * 0.3 + loss_object(res_origins_, origins) * 0.3

      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      train_loss(loss)

    maps = create_grid(im_sz)
    maps = np.concatenate((maps, origins * 0.1, targets * 0.1), axis=-1).astype(np.float32)

    epoch = 0
    template = 'Epoch {}, Loss: {}'

    t = tqdm(range(train_epochs), desc=template.format(epoch, train_loss.result()))

    for i in t:
        epoch = i + 1

        t.set_description(template.format(epoch, train_loss.result()))
        t.refresh()

        train_step(maps, origins, targets, warp_scale, add_scale, mult_scale, add_first, im_sz)

        if (epoch < 100 and epoch % 10 == 0) or\
           (epoch < 1000 and epoch % 100 == 0) or\
           (epoch % 1000 == 0):
            preds = model(maps, training=False)[:1]
            preds = tf.image.resize(preds, [im_sz, im_sz])
            np.save(output_name, preds.numpy())

def plot_maps(path_preds, channel_ranges = [(0, 3), (3, 6), (6, 8), (8, 11), (11, 14), (14, 16)]):
    
    preds = np.load(path_preds)

    # Define the number of subplots
    num_subplots = len(channel_ranges)  # Adjust as needed

    # Create a figure and axis
    fig, axes = plt.subplots(1, num_subplots, figsize=(15, 3))  # Adjust figsize as needed

    # Iterate through the specified ranges and create subplots
    for i in range(num_subplots):
        start_channel, end_channel = channel_ranges[i]

        # Extract the specified slice of the array
        subplot_data = preds[0, :, :, start_channel:end_channel]

        if subplot_data.shape[-1] == 2:
            subplot_data = subplot_data[:, :, 0]  # Choose the first dimension, you can change this as needed

        subplot_data = np.clip(subplot_data, -1, 1)
        subplot_data = ((subplot_data + 1) * 127.5).astype(np.uint8)

        # Plot the data on the corresponding subplot
        
        axes[i].imshow(subplot_data, cmap='rainbow')  # Adjust cmap as needed
        axes[i].axis('off')

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

def morphing(source, 
             target, 
             path_preds, 
             steps = 100, 
             warp_scale = 0.05, 
             add_scale = 0.4, 
             mult_scale = 0.4, 
             add_first = False,
             im_sz = 1024, 
             each = 10):
    
    dom_a = cv2.imread(source, cv2.IMREAD_COLOR)
    dom_b = cv2.imread(target, cv2.IMREAD_COLOR)

    # Checks if input and destination image are of the same dimensions.
    if (dom_a.shape[1] != dom_b.shape[1] or dom_a.shape[0] != dom_b.shape[0]):
        print("Input Image is not the same dimensions as Destination Image.")
        sys.exit()

    dom_a = cv2.cvtColor(dom_a, cv2.COLOR_BGR2RGB)
    dom_a = cv2.resize(dom_a, (im_sz, im_sz), interpolation = cv2.INTER_AREA)
    dom_a = dom_a / 127.5 - 1


    dom_b = cv2.cvtColor(dom_b, cv2.COLOR_BGR2RGB)
    dom_b = cv2.resize(dom_b, (im_sz, im_sz), interpolation = cv2.INTER_AREA)
    dom_b = dom_b / 127.5 - 1

    origins = dom_a.reshape(1, im_sz, im_sz, 3).astype(np.float32)
    targets = dom_b.reshape(1, im_sz, im_sz, 3).astype(np.float32)


    preds = np.load(path_preds)

    org_strength = tf.reshape(tf.range(steps, dtype=tf.float32), [steps, 1, 1, 1]) / (steps - 1)
    trg_strength = tf.reverse(org_strength, axis = [0])

    fig, axes = plt.subplots(1, steps // each, figsize=(20, 3))

    k = 0
    for i in range(steps):
        preds_org = preds * org_strength[i]
        preds_trg = preds * trg_strength[i]

        res_targets, res_origins = warp(origins, targets, preds_org[...,:8], preds_trg[...,8:], add_scale, mult_scale, im_sz, warp_scale, add_first)
        res_targets = tf.clip_by_value(res_targets, -1, 1)
        res_origins = tf.clip_by_value(res_origins, -1, 1)

        results = res_targets * trg_strength[i] + res_origins * org_strength[i]
        res_numpy = results.numpy()

        img = ((res_numpy[0] + 1) * 127.5).astype(np.uint8)

        if (i+1) % each == 0:
            axes[k].imshow(img)  
            axes[k].axis('off')
            k += 1
    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
    
