#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np


# In[ ]:


def models_accuracy(resnet_model, ViT_L, ViT_preprocess, image_folder, text_inputs, target_index, labels):
    # Transformation for input images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Get the list of image files in the folder
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    overall_resnet_accuracies = []
    overall_vit_l_accuracies = []
    plt.figure(figsize=(20, 2))
    # Process all images in the folder
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        image_input = ViT_preprocess(image).unsqueeze(0)

        # Calculate features using ViT-L
        with torch.no_grad():
            image_features = ViT_L.encode_image(image_input)
            text_features = ViT_L.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        vit_l_predicted = indices

        image_tensor = transform(image).unsqueeze(0)
        target_label = torch.tensor([target_index])

        # Make predictions using ResNet-101
        with torch.no_grad():
            resnet_outputs = resnet_model(image_tensor)
        _, resnet_predicted = resnet_outputs.max(1)
        resnet_predicted += 1
        # Calculate accuracies
        resnet_accuracy = (resnet_predicted == target_label).item() * 100
        vit_l_accuracy = (vit_l_predicted == target_label).item() * 100

        overall_resnet_accuracies.append(resnet_accuracy)
        overall_vit_l_accuracies.append(vit_l_accuracy)

        # Display images and predictions in a subplot
        plt.subplot(1, len(image_files) + 1, i + 1)
        image_np = image_tensor[0].permute(1, 2, 0)
        image_np = (image_np.clamp(-1, 1) + 1) / 2.0  # normalization to 0..1 range
        plt.imshow(image_np)
        plt.axis("off")
        results = f"ResNet: {labels[resnet_predicted.item()]}\n"
        results += f"ViT-L: {labels[vit_l_predicted.item()]}"
        plt.title(results)

    # Display bar plot for both models
    plt.subplot(1, len(image_files) + 1, i + 2)
    plt.barh([1, 2], [np.mean(overall_resnet_accuracies), 
                      np.mean(overall_vit_l_accuracies)], 
             color=['gray', 'blue'])
    plt.yticks([1, 2], ['ResNet', 'ViT-L'])
    plt.xlim([0, 100])
    plt.show()

    # Calculate overall accuracies
    overall_resnet_accuracy = np.mean(overall_resnet_accuracies)
    overall_vit_l_accuracy = np.mean(overall_vit_l_accuracies)
