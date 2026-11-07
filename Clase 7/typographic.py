#!/usr/bin/env python
# coding: utf-8

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

def typographic_attack(model, preprocess, folder, text_inputs, labels, k = 5, font_size = 12):
    # Folder containing images
    folder_path = f"{folder}"
    image_files = os.listdir(folder_path)
    image_files.sort(reverse=True)
    fig, axs = plt.subplots(2, len(image_files), figsize=(4*len(image_files), 8))

    for i, image_file in enumerate(image_files):
        # Load and preprocess the image
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Pick the top k most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(k)

        label_names = []
        probs = []

        # Print the result
        for value, index in zip(values, indices):
            name = labels[index]
            y_proba = 100 * value.item()
            label_names.append(name)
            probs.append(y_proba)

        # Display the image in the first row
        axs[0, i].imshow(image)
        axs[0, i].axis('off')

        # Create a table with labels and probabilities in the second row
        table_data = {'Label': label_names, 
                      'Probability (%)': [f"{round(prob, 2)}%" for prob in probs]}
        table = axs[1, i].table(cellText=[list(map(str, row)) 
                                          for row in zip(table_data['Label'], 
                                                         table_data['Probability (%)'])],
                                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

        axs[1, i].axis('off')
        # Increase the font size in the table
        table.auto_set_font_size(False)
        table.set_fontsize(font_size) 

    plt.tight_layout(pad=0.5)
    plt.show()

