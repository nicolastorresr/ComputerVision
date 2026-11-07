#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import torch
from torchvision import models, transforms
import copy
import torch.optim as optim
import numpy as np
from lucent.misc.io import show

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


# Utils
# Load image file
def load_image(path):
    # Images loaded as BGR
    img = cv2.imread(path)
    return img

# Convert image
def convert(img):
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # imshow() only accepts float [0,1] or int [0,255]
    img = np.array(img/255).clip(0,1)
    
    return img
    
# Save Image as out{num_iterms}.png
def saveimg(img, iters, PIXEL_CLIP = 'True', path='out'):
    if (PIXEL_CLIP=='True'):
        img = img.clip(0, 255)
    cv2.imwrite(path + '/out'+str(iters)+'.png', img)
    
# Color transfer
def transfer_color(src, dest, PIXEL_CLIP = 'True'):
    if (PIXEL_CLIP=='True'):
        src, dest = src.clip(0,255), dest.clip(0,255)
        
    # Resize src to dest's size
    H,W,_ = src.shape 
    dest = cv2.resize(dest, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
    
    dest_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY) #1 Extract the Destination's luminance
    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)   #2 Convert the Source from BGR to YIQ/YCbCr
    src_yiq[...,0] = dest_gray                         #3 Combine Destination's luminance and Source's IQ/CbCr
    
    return cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR)  #4 Convert new image from YIQ back to BGR


# In[ ]:


# Preprocessing
def itot(img, MAX_IMAGE_SIZE = 512):
    # Rescale the image
    H, W, C = img.shape
    image_size = tuple([int((float(MAX_IMAGE_SIZE) / max([H,W]))*x) for x in [H, W]])
    
    itot_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    
    # Subtract the means
    normalize_t = transforms.Normalize([103.939, 116.779, 123.68],[1,1,1])
    tensor = normalize_t(itot_t(img)*255)
    
    # Add the batch_size dimension
    tensor = tensor.unsqueeze(dim=0)
    return tensor

def ttoi(tensor):
    # Add the means
    ttoi_t = transforms.Compose([
        transforms.Normalize([-103.939, -116.779, -123.68],[1,1,1])])
    
    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    img = ttoi_t(tensor)
    img = img.cpu().numpy()
    
    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img


# In[ ]:


# Load VGG19 Skeleton
vgg = models.vgg19(weights=None)

# Load pretrained weights
vgg.load_state_dict(torch.load('weights/vgg19-d01eb7cb.pth'), strict=False)

# Change Pooling Layer
def pool_(model, pool='avg'):
    if (pool=='avg'):
        ct=0
        for layer in model.children():
            if isinstance(layer, torch.nn.MaxPool2d):
                model[ct] = torch.nn.AvgPool2d(kernel_size=2, 
                                               stride=2, 
                                               padding=0, 
                                               ceil_mode=False)
            ct+=1
    elif (pool=='max'):
        ct=0
        for layer in model.children():
            if isinstance(layer, torch.nn.AvgPool2d):
                model[ct] = torch.nn.MaxPool2d(kernel_size=2, 
                                               stride=2, 
                                               padding=0, 
                                               dilation=1, 
                                               ceil_mode=False)
            ct+=1
            
    return model


# Extract only the 'features' network, 
# 'classifier' network is not needed
model = copy.deepcopy(vgg.features)
model.to(device)

# Turn-off unnecessary gradient tracking
for param in model.parameters():
    param.requires_grad = False


# In[ ]:


"""
Loss Functions
All loss functions are basically Mean of the Squared Errors (MSE)
g - generated features
c - content features
s - style features

1. What is Total Variation(TV) Loss ???
    The total variation (TV) loss encourages spatial smoothness in the generated image. 
   It was not used by Gatys et al in their CVPR paper but it can sometimes improve the 
   results; for more details and explanation see Mahendran and Vedaldi 
   "Understanding Deep Image Representations by Inverting Them" CVPR 2015.

   - @jcjohnson
   https://github.com/jcjohnson/neural-style/issues/302

2. How to implement TV Loss?
    https://en.wikipedia.org/wiki/Total_variation_denoising
"""
mse_loss = torch.nn.MSELoss()

def gram(tensor):
    B, C, H, W = tensor.shape
    x = tensor.view(C, H*W)
    return torch.mm(x, x.t())

def content_loss(g, c):
    loss = mse_loss(g, c)
    return loss
    
def style_loss(g, s):
    c1,c2 = g.shape
    loss = mse_loss(g, s)
    return loss / (c1**2) # Divide by square of channels

def tv_loss(c):
    x = c[:,:,1:,:] - c[:,:,:-1,:]
    y = c[:,:,:,1:] - c[:,:,:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return loss


# In[ ]:


# VGG Forward Pass
def get_features(model, tensor):
    layers = {
        '3': 'relu1_2',   # Style layers
        '8': 'relu2_2',
        '17' : 'relu3_3',
        '26' : 'relu4_3',
        '35' : 'relu5_3',
        '22' : 'relu4_2', # Content layers
        #'31' : 'relu5_2'
    }
    
    # Get features
    features = {}
    x = tensor
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            if (name=='22'):   # relu4_2
                features[layers[name]] = x
            elif (name=='31'): # relu5_2
                features[layers[name]] = x
            else:
                b, c, h, w = x.shape
                features[layers[name]] = gram(x) / (h*w)
                
            # Terminate forward pass
            if (name == '35'):
                break
            
    return features


# In[ ]:


# Generate Initial Image
def initial(content_tensor, init_image='random'):
    B, C, H, W = content_tensor.shape
    if (init_image=='random'):
        #tensor = torch.randn(C, H, W).mul(torch.std(content_tensor.clone().cpu())/255).unsqueeze(0)
        tensor = torch.randn(C, H, W).mul(0.001).unsqueeze(0)
    else:
        tensor = content_tensor.clone().detach()
    
    return tensor




# In[ ]:


def stylize(content_tensor,
            style_tensor,
            g,
            OPTIMIZER = 'adam',
            ADAM_LR = 10,
            PRESERVE_COLOR = "False", 
            CONTENT_WEIGHT = 5e0,
            STYLE_WEIGHT = 1e2,
            TV_WEIGHT = 1e-3,
            NUM_ITER = 500, 
            SHOW_ITER = 100, 
            SAVE = False, 
            PATH = "out"):     
    # Get features representations/Forward pass
    content_layers = ['relu4_2']
    content_weights = {'relu4_2': 1.0} 
    style_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
    style_weights = {'relu1_2': 0.2, 'relu2_2': 0.2, 'relu3_3': 0.2, 'relu4_3': 0.2, 'relu5_3': 0.2}
    c_feat = get_features(model, content_tensor)
    s_feat = get_features(model, style_tensor)
    
    """
    Define Optimizer
    The optimizer minimizes the total loss by updating the tensor 'g'.
    """
    if (OPTIMIZER=='lbfgs'):
        optimizer = optim.LBFGS([g])
    else:
        optimizer = optim.Adam([g], lr=ADAM_LR)
    
    list_of_images = [[],[]]
    
    i = [0]
    while i[0] < NUM_ITER:
        def closure(PRESERVE_COLOR = PRESERVE_COLOR, 
            CONTENT_WEIGHT = CONTENT_WEIGHT,
            STYLE_WEIGHT = STYLE_WEIGHT,
            TV_WEIGHT = TV_WEIGHT,
            NUM_ITER = NUM_ITER, 
            SHOW_ITER = SHOW_ITER, 
            SAVE = SAVE, 
            PATH = PATH):
            # Zero-out gradients
            optimizer.zero_grad()

            # Forward pass
            g_feat = get_features(model, g)

            # Compute Losses
            c_loss=0
            s_loss=0
            for j in content_layers:
                c_loss += content_weights[j] * content_loss(g_feat[j], c_feat[j])
            for j in style_layers:
                s_loss += style_weights[j] * style_loss(g_feat[j], s_feat[j])
            
            c_loss = CONTENT_WEIGHT * c_loss
            s_loss = STYLE_WEIGHT * s_loss
            t_loss = TV_WEIGHT * tv_loss(g.clone().detach())
            total_loss = c_loss + s_loss + t_loss

            # Backprop
            total_loss.backward(retain_graph=True)
            
            if ((i[0] % SHOW_ITER) == 0):

                if (PRESERVE_COLOR=='True'):
                    g_ = transfer_color(ttoi(content_tensor.clone().detach()), ttoi(g.clone().detach()))
                else:
                    g_ = ttoi(g.clone().detach())
                
                list_of_images[0].append(convert(g_))
                list_of_images[1].append("Iteration: {}".format(i[0]))

                if SAVE:
                    saveimg(g_, i[0]-1, PATH)
            i[0]+=1
            
            return (total_loss)
        
        # Weight/Pixel update
        optimizer.step(closure)
    show(list_of_images[0], labels=list_of_images[1])


# In[ ]:




