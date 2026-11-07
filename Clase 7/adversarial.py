#!/usr/bin/env python
# coding: utf-8
import os
import json
import numpy as np 
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision import transforms

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "saved"
# Name of the model
MODEL_NAME = "inceptionv1"

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    
# Mean and Std from ImageNet
NORM_MEAN = np.array([0.5, 0.5, 0.5])
NORM_STD = np.array([0.5, 0.5, 0.5])


def show_prediction(img, label, label_names, pred, K=5, adv_img=None, noise=None, epsilon=None):
    
    if isinstance(img, torch.Tensor):
        # Tensor image to numpy
        img = img.cpu().permute(1, 2, 0).numpy()
        img = (img * NORM_STD[None,None]) + NORM_MEAN[None,None]
        img = np.clip(img, a_min=0.0, a_max=1.0)
        label = label.item()
    
    # Plot on the left the image with the true label as title.
    # On the right, have a horizontal bar plot with the top k predictions including probabilities
    if noise is None or adv_img is None:
        fig, ax = plt.subplots(1, 2, figsize=(12,2), gridspec_kw={'width_ratios': [1, 1]})
    else:
        fig, ax = plt.subplots(1, 5, figsize=(16,2), gridspec_kw={'width_ratios': [1, 1, 1, 1, 2]})
    
    ax[0].imshow(img)
    ax[0].set_title(label_names[label])
    ax[0].axis('off')
    
    if adv_img is not None and noise is not None:
        # Visualize adversarial images
        adv_img = adv_img.cpu().permute(1, 2, 0).numpy()
        adv_img = (adv_img * NORM_STD[None,None]) + NORM_MEAN[None,None]
        adv_img = np.clip(adv_img, a_min=0.0, a_max=1.0)
        ax[2].imshow(adv_img)
        ax[2].set_title('Adversarial')
        ax[2].axis('off')
        # Visualize noise
        noise = noise.cpu().permute(1, 2, 0).numpy()
        noise = noise * 0.5 + 0.5 # Scale between 0 to 1 
        ax[1].imshow(noise)
        ax[1].set_title(f'Noise ($ϵ=${epsilon})')
        ax[1].axis('off')
        # buffer
        ax[3].axis('off')
    
    if abs(pred.sum().item() - 1.0) > 1e-4:
        pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(K, dim=-1)
    topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
    ax[-1].barh(np.arange(K), topk_vals*100.0, align='center', 
                color = ["tomato" if i == 0 and topk_idx[i] != label else "lightgray" if topk_idx[i] != label 
                         else "lightgreen" for i in range(K)])
    ax[-1].set_yticks(np.arange(K))
    ax[-1].set_yticklabels([label_names[c] for c in topk_idx])
    ax[-1].invert_yaxis()
    ax[-1].set_xlabel('Confidence')
    ax[-1].set_title('Predictions')
    
    plt.show()
    plt.close()


# In[ ]:


def fast_gradient_sign_method(model, imgs, labels, epsilon=0.02):
    # Determine prediction of the model
    inp_imgs = imgs.clone().requires_grad_()
    preds = model(inp_imgs.to(device))
    preds = F.log_softmax(preds, dim=-1)
    # Calculate loss by NLL
    loss = -torch.gather(preds, 1, labels.to(device).unsqueeze(dim=-1))
    loss.sum().backward()
    # Update image to adversarial example as written above
    noise_grad = torch.sign(inp_imgs.grad.to(imgs.device))
    fake_imgs = imgs + epsilon * noise_grad
    fake_imgs.detach_()
    return fake_imgs, noise_grad


# In[ ]:


def adversarial_attack_fgsm(model, exmp_batch, label_batch, label_names, idx, epsilon=0.02):
    adv_imgs, noise_grad = fast_gradient_sign_method(model, exmp_batch, label_batch, epsilon)
    with torch.no_grad():
        adv_preds = model(adv_imgs.to(device))
    show_prediction(exmp_batch[idx], 
                    label_batch[idx], 
                    label_names,
                    adv_preds[idx], 
                    adv_img=adv_imgs[idx], 
                    noise=noise_grad[idx], 
                    epsilon=epsilon)


# In[ ]:


def place_patch(img, patch):
    for i in range(img.shape[0]):
        h_offset = np.random.randint(0,img.shape[2]-patch.shape[1]-1)
        w_offset = np.random.randint(0,img.shape[3]-patch.shape[2]-1)
        img[i,:,h_offset:h_offset+patch.shape[1],w_offset:w_offset+patch.shape[2]] = patch_forward(patch)
    return img


TENSOR_MEANS, TENSOR_STD = torch.FloatTensor(NORM_MEAN)[:,None,None], torch.FloatTensor(NORM_STD)[:,None,None]
def patch_forward(patch):
    # Map patch values from [-infty,infty] to ImageNet min and max
    patch = (torch.tanh(patch) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)
    return patch


def eval_patch(model, patch, val_loader, target_class):
    model.eval()
    tp, tp_5, counter = 0., 0., 0.
    with torch.no_grad():
        for img, img_labels in tqdm(val_loader, desc="Validating...", leave=False):
            # For stability, place the patch at 4 random locations per image, and average the performance
            for _ in range(4): 
                patch_img = place_patch(img, patch)
                patch_img = patch_img.to(device)
                img_labels = img_labels.to(device)
                pred = model(patch_img)
                # In the accuracy calculation, we need to exclude the images that are of our target class
                # as we would not "fool" the model into predicting those
                tp += torch.logical_and(pred.argmax(dim=-1) == target_class, img_labels != target_class).sum()
                tp_5 += torch.logical_and((pred.topk(5, dim=-1)[1] == target_class).any(dim=-1), 
                                          img_labels != target_class).sum()
                counter += (img_labels != target_class).sum()
    acc = tp/counter
    top5 = tp_5/counter
    return acc, top5


# In[ ]:


def patch_attack(model, target_class, patch_size=64, num_epochs=5):
    # Leave a small set of images out to check generalization
    # In most of our experiments, the performance on the hold-out data points
    # was as good as on the training set. Overfitting was little possible due
    # to the small size of the patches.
    train_set, val_set = torch.utils.data.random_split(dataset, [4500, 500])
    train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    
    # Create parameter and optimizer
    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size)
    patch = nn.Parameter(torch.zeros(3, patch_size[0], patch_size[1]), requires_grad=True)
    optimizer = torch.optim.SGD([patch], lr=1e-1, momentum=0.8)
    loss_module = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        t = tqdm(train_loader, leave=False)
        for img, _ in t:
            img = place_patch(img, patch)
            img = img.to(device)
            pred = model(img)
            labels = torch.zeros(img.shape[0], device=pred.device, dtype=torch.long).fill_(target_class)
            loss = loss_module(pred, labels)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            t.set_description(f"Epoch {epoch}, Loss: {loss.item():4.2f}")
    
    # Final validation
    acc, top5 = eval_patch(model, patch, val_loader, target_class)
    
    return patch.data, {"acc": acc.item(), "top5": top5.item()}


# In[ ]:


# If you train new patches, you can save the results via calling this function
def save_results(patch_dict):
    result_dict = {cname: {psize: [t.item() if isinstance(t, torch.Tensor) else t 
                                   for t in patch_dict[cname][psize]["results"]] 
                           for psize in patch_dict[cname]} 
                   for cname in patch_dict}
    with open(f"{CHECKPOINT_PATH}/patches_{MODEL_NAME}/patch_results.json", "w") as f:
        json.dump(result_dict, f, indent=4)


# In[ ]:


def get_patches(model, data, class_names, patch_sizes, label_names, json_results):
    result_dict = dict()

    # Loop over all classes and patch sizes
    for name in class_names:
        result_dict[name] = dict()
        for patch_size in patch_sizes:
            c = label_names.index(name)            
            file_name = f"{CHECKPOINT_PATH}/patches_{MODEL_NAME}/{name}_{patch_size}_patch.pt"
            # Load patch if pretrained file exists, otherwise start training
            if not os.path.isfile(file_name):
                print(f"Weights for {name} size {patch_size} do not exist, start training.")
                patch, val_results = patch_attack(model, target_class=c, patch_size=patch_size, num_epochs=5)
                print(f"Validation results for {name} and {patch_size}:", val_results)
                torch.save(patch, file_name)
            else:
                print(f"Loaded weights for {name} size {patch_size}.")
                patch = torch.load(file_name)
            # Load evaluation results if exist, otherwise manually evaluate the patch
            if name in json_results:
                results = json_results[name][str(patch_size)]
            else:
                results = eval_patch(model, patch, data, target_class=c)    
            
            # Store results and the patches in a dict for better access
            result_dict[name][patch_size] = {
                "results": results,
                "patch": patch
            }
        
    return result_dict


# In[ ]:


def show_patches(class_names, patch_sizes, patch_dict):
    fig, ax = plt.subplots(len(class_names), len(patch_sizes), figsize=(len(patch_sizes)*2.2, len(class_names)*2.2))
    for c_idx, cname in enumerate(class_names):
        for p_idx, psize in enumerate(patch_sizes):
            patch = patch_dict[cname][psize]["patch"]
            patch = (torch.tanh(patch) + 1) / 2 # Parameter to pixel values
            patch = patch.cpu().permute(1, 2, 0).numpy()
            patch = np.clip(patch, a_min=0.0, a_max=1.0)
            ax[p_idx].imshow(patch)
            ax[p_idx].set_title(f"{cname}, size {psize}")
            ax[p_idx].axis('off')
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()


# In[ ]:


import tabulate
from IPython.display import display, HTML

def show_table(patch_dict, class_names, psize):    
    table = [[name] + [f"{(100.0 * patch_dict[name][psize]['results'][i]):4.2f}%" for i in [0,1]]
             for name in class_names]
    display(HTML(tabulate.tabulate(table, 
                                   tablefmt='html', 
                                   headers=["Class name"] + [f"Top-{n} Patch size {psize}" for n in [1,5]])))


# In[ ]:


def perform_patch_attack(patch,model,exmp_batch,label_batch,label_names,idx):
    patch_batch = exmp_batch.clone()
    patch_batch = place_patch(patch_batch, patch)
    with torch.no_grad():
        patch_preds = model(patch_batch.to(device))
    show_prediction(patch_batch[idx], label_batch[idx], label_names, patch_preds[idx])


# In[ ]:

def targeted_adversary(image, epsilon, gradient):
    
    # Similar a lo anterior, pero ahora estamos minimizando el error!
    perturbed_image = image - epsilon * gradient
    
    # Asegura que los valores de los píxeles permanezcan dentro del rango válido [0,1].
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
  
    return perturbed_image


def create_adversarial_examples(data, model, class_idx, target_class, epsilon = 0.1, k = 3):
    
    batch_size = len(data)

    t = [target_class] * batch_size
    example_found = [False] * batch_size
    original_examples = [None] * batch_size
    adversarial_examples = [None] * batch_size
    prediction_names = [[None]*k] * batch_size

    num_found = 0

    # Loop over all examples in test set
    count = 0
    output = None
    
    target = torch.LongTensor(t)
    while num_found < len(t):
        count += 1
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero  gradients
        model.zero_grad()

        # Calculate gradient
        loss.backward()

        data_grad = data.grad.data

        # Attempt to create a targeted adversary
        perturbed_data = targeted_adversary(data, epsilon, data.grad)

        if not output is None:
            predictions = torch.max(output.data, 1)[1].cpu().numpy()            
            top_k_values, top_k_indices = torch.topk(output, k)

            for x in range(len(predictions)):
                if example_found[x] != True and predictions[x] == target_class:
                    adversarial_examples[x] = data[x].cpu().detach().permute(1, 2, 0)
                    example_found[x] = True
                   
                    indices = top_k_indices[x].cpu().numpy()[::-1]
                    values = top_k_values[x].detach().numpy()[::-1]                                       
                    class_names = [class_idx[str(idx)][1] for idx in indices]

                    for y in range(len(class_names)):
                        prediction_names[x][y] = (class_names[y], values[y])

                    num_found = 0
                    for found in example_found:
                        if found:
                            num_found += 1                    

        data = perturbed_data.clone().detach()

        # Re-classify the perturbed image
        output = model(perturbed_data)
        
    return prediction_names, adversarial_examples


# In[ ]:


def adversarial_prediction(data, idx, model, class_idx, prediction_names, adversarial_examples):    
    
    fig, ax = plt.subplots(1, 4, figsize=(11,2), gridspec_kw={'width_ratios': [1, 1, 1, 2]})
    
    original_image = data[idx].cpu().detach().permute(1, 2, 0) 
       
    label = class_idx[str(torch.max(model(data).data, 1)[1].cpu().numpy()[idx])][1]
    
    # Original Image
    ax[0].set_title(label)
    ax[0].imshow(original_image)
    ax[0].axis('off')

    # Perturbed Image
    ax[1].set_title('Adversarial')
    ax[1].imshow(adversarial_examples[idx])
    ax[1].axis('off')
    
    # buffer
    ax[2].axis('off')
        
    # Predictions
    labels, values = zip(*prediction_names[idx])
    ax[3].barh(labels, values, 
               color = ["tomato" if i == len(labels)-1 and labels[i] != label 
                        else "lightgray" if labels[i] != label 
                        else "lightgreen" for i in range(len(labels))])
    ax[3].set_xlabel('Confidence')
    ax[3].set_title('Predictions')

