#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# In[24]:


import keras
import tensorflow as tf


# In[9]:


from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.utils import load_img
from keras.utils import img_to_array
from numpy import expand_dims


# In[30]:


from matplotlib import colormaps


# In[264]:


def find_last_conv_layer(model):
    layer_names = [layer.name for layer in model.layers][::-1]
    for x in layer_names:
        if "conv" in x:
            return x
    return layer_names[-1]


# In[260]:


# Takes two images and composites them
def composite_img(back_path, 
                  patch_path, 
                  size=(299,299), 
                  patch_width_ratio=0.1, 
                  patch_position=(0.0, 0.0)):
    
    background_img = load_img(back_path)
    patch_img = load_img(patch_path)
    
    composite = background_img.copy()
    composite = np.array(composite, np.float64)
    composite = composite / 255
    patch = patch_img.copy()
    patch = np.array(patch, np.float64)
    patch = patch / 255
    
    composite_img = Image.fromarray(np.uint8(composite * 255.))
    composite_img = composite_img.resize(size, Image.LANCZOS)
    composite = (np.asarray(composite_img)/255.)[:,:,:3]

    if patch_width_ratio > 0: 
        patch_img_width = int(patch_width_ratio * size[0])
        patch_img_ = Image.fromarray(np.uint8(patch[:,:,:3] * 255.))
        patch_img_thumb = patch_img_.resize((patch_img_width, patch_img_width), Image.LANCZOS)
        patch_img_thumb_array = np.asarray(patch_img_thumb)/255.
        patch_img_thumb_size = patch_img_thumb_array.shape

        patch_y = int(patch_position[1] * (size[1] - patch_img_thumb_size[1]))
        patch_x = int(patch_position[0] * (size[0] - patch_img_thumb_size[0]))
        
        x_1 = patch_y + patch_img_thumb_size[1]
        x_2 = patch_x + patch_img_thumb_size[0]

        composite[patch_y:x_1, patch_x:x_2, 0:3] = patch_img_thumb_array

    composite_img = Image.fromarray(np.uint8(composite * 255.))
    composite_img_resize = composite_img.resize(size, Image.LANCZOS)
    composite_img_resize_array = np.asarray(composite_img_resize)
    return composite_img_resize_array


# In[151]:


def get_gradcam(img, heatmap, alpha=0.4, cmap='jet'):
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    cm = colormaps[cmap]

    # Use RGB values of the colormap
    cm_colors = cm(np.arange(256))[:, :3]
    cm_heatmap = cm_colors[heatmap]

    # Create an image with RGB colorized heatmap
    cm_heatmap = keras.utils.array_to_img(cm_heatmap)
    cm_heatmap = cm_heatmap.resize((img.shape[1], img.shape[0]))
    cm_heatmap = keras.utils.img_to_array(cm_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = cm_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    
    return (superimposed_img)


# In[152]:


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# In[195]:


def get_index_for_class(class_name, data):
    for index, values in data.items():
        if values[1] == class_name:
            return int(index)
    return None  # Return None if the class name is not found


# In[238]:


def classify(image, 
             model, 
             class_idx,
             n=3, 
             show_grad=False, 
             last_conv_layer_name=None, 
             class_name=None,
             alpha=1, 
             cmap='Greys_r'):
    rgb_gray = (0.827, 0.827, 0.827)
    image_array = img_to_array(image)
    
    processed_image = preprocess_input(image_array)  
    expanded_image = expand_dims(processed_image, axis=0) 

    predictions = model.predict(expanded_image, verbose=0)
    top_K = decode_predictions(predictions, top=n)[0]
    
    label_names = []
    probs = []
    clrs = []
    
    for _, name, y_proba in top_K:
        label_names.append(name)
        probs.append(y_proba)
        if class_name is None:
            clrs.append(rgb_gray)
        else:
            if name == class_name:
                # Use jet colormap to colorize heatmap
                cm = colormaps.get_cmap(cmap)
                # Use RGB values of the colormap
                rgb = cm(0.9)[:-1]
                clrs.append(rgb)
            else:
                clrs.append(rgb_gray)
            
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if show_grad:   
        # Convert back to the original range [0, 255]
        original_matrix = (image_array + 1) * 127.5
        # Round and cast to uint8 for pixel values
        original_matrix_int = np.round(original_matrix).astype(np.uint8)
        index_number = get_index_for_class(class_name, class_idx)
        if last_conv_layer_name is None:
            last_conv_layer_name = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(expanded_image, model, last_conv_layer_name, pred_index=index_number)
        superimposed_img = get_gradcam(original_matrix_int, heatmap, alpha=alpha, cmap=cmap)
        ax1.imshow(superimposed_img)
        ax1.axis('off')
        if class_name is None:
            ax1.set_title("Class activation heatmap")
        else:
            ax1.set_title("Class activation heatmap for {}".format(class_name))
    else:
        # Display image on the left subplot
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Input Image')

    # Create bar plot on the right subplot
    ax2.barh(label_names, probs, color=clrs)
    ax2.set_xlabel('Confidence')
    ax2.set_title(f'Top-{n} Predictions')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.show()


# In[ ]:




