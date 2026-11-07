#!/usr/bin/env python
# coding: utf-8
# pip install -U ultralytics
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_input_vgg16, decode_predictions as decode_predictions_vgg16
from keras.applications.resnet import ResNet50, preprocess_input as preprocess_input_resnet50, decode_predictions as decode_predictions_resnet50
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_input_inceptionv3, decode_predictions as decode_predictions_inceptionv3

def recognition(model_names, model_vgg16, model_resnet50, model_inceptionv3, filename, K=10):
    # Load the image
    image = skimage.io.imread(filename)
    
    # Create a dictionary to hold predictions for each model
    predictions_dict = {}
    
    for model_name in model_names:
        # Get the input shape for the specified model
        height, width = model_names[model_name]
        image_resize = skimage.transform.resize(image, (height, width), preserve_range=True)
        x = np.expand_dims(image_resize, axis=0)
        
        # Choose the model based on the input model_name
        if model_name == 'vgg16':
            model = model_vgg16
            preprocess = preprocess_input_vgg16
            decode_predictions = decode_predictions_vgg16
        elif model_name == 'resnet50':
            model = model_resnet50
            preprocess = preprocess_input_resnet50
            decode_predictions = decode_predictions_resnet50
        elif model_name == 'inceptionv3':
            model = model_inceptionv3
            preprocess = preprocess_input_inceptionv3
            decode_predictions = decode_predictions_inceptionv3
        else:
            print(f"Invalid model name: {model_name}")
            return
        
        # Preprocess the image
        x = preprocess(x)
        
        # Get predictions
        predictions = model.predict(x, verbose=0)
        top_K = decode_predictions(predictions, top=K)[0]
        
        # Store predictions in the dictionary
        predictions_dict[model_name] = top_K
    
    # Display the image
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 2]})
    axs[0].imshow(image)
    axs[0].axis('off')

    # Display predictions in a table
    headers = ['Rank', 'VGG16', 'Score', 'ResNet50', 'Score', 'InceptionV3', 'Score']
    table_data = []

    for i in range(K):
        vgg_pred = predictions_dict['vgg16'][i]
        resnet_pred = predictions_dict['resnet50'][i]
        inception_pred = predictions_dict['inceptionv3'][i]

        vgg_pred_score = str(round(vgg_pred[2]*100,2))+"%"
        resnet_pred_score = str(round(resnet_pred[2]*100,2))+"%"
        inception_pred_score = str(round(inception_pred[2]*100,2))+"%"

        table_data.append([i+1, vgg_pred[1], vgg_pred_score, 
                           resnet_pred[1], resnet_pred_score, 
                           inception_pred[1], inception_pred_score])

    axs[1].axis('tight')
    axs[1].axis('off')
    col_widths = [0.05, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1]
    table = axs[1].table(cellText=table_data, loc='center', 
                         cellLoc='center', colLabels=headers, colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    plt.tight_layout(pad=0.5)
    table.scale(1, 2) 
    plt.show()

from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from torchvision.io import read_image

def get_colors(n_classes):
    palette = np.random.uniform(0, 255, size=(n_classes, 3))
    colors = torch.as_tensor([i for i in range(n_classes)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def deeplab_segmentation(model, filename):
    colors = get_colors(21) # Number of classes acording COCO_train2017
    input_image = Image.open(filename)    
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)    
    
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)    
    
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(121)   
    plt.imshow(input_image)
    plt.axis('off')
    
    ax2 = fig.add_subplot(122)
    plt.imshow(r)
    plt.axis('off')
    plt.show()
    
def plot_result(model, path, fig_size=(12,8)):
    result = model(path)
    fig = plt.figure(figsize=fig_size)
    plt.imshow(result.render()[0])
    plt.axis('off')
    plt.show()

import cv2
import copy

from ultralytics import YOLO, YOLOWorld

def plot_yolo_world(path, classes=None, verbose=False, fig_size=(12,8)):
    model = YOLOWorld('yolov8s-world.pt')
    if classes is not None:
        model.set_classes(classes)
    result = model.predict(path, verbose=verbose)
    image = result[0].plot()
    fig = plt.figure(figsize=fig_size)
    color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(color)
    plt.axis('off')
    plt.show()

def process_image(filename,height=500,width=500):
    image = read_image(filename)
    return [transforms.Resize(size=(height,width), antialias=True)(image)]

def filter_model_output(output,score_threshold = .8):
    filtred_output = list()
    for image in output:
        filtred_image = dict()
        for key in image.keys():
            filtred_image[key] = image[key][image['scores'] >= score_threshold]
        filtred_output.append(filtred_image)
    return filtred_output

def get_boolean_mask(output):
    for index,pred in enumerate(output):
        output[index]['masks'] = pred['masks'] > 0.5
        output[index]['masks'] = output[index]['masks'].squeeze(1)
    return output


def mask_segmentation(model,weights,filename,class_names,score_threshold=0.8,height=500,width=500):
    image_list = process_image(filename,height,width)
    transforms = weights.transforms()
    images = [transforms(d) for d in image_list]
    output = model(images)
    output = filter_model_output(output=output,score_threshold=score_threshold)
    output = get_boolean_mask(output)
    
    for index, (image, prediction) in enumerate(zip(image_list, output)):
        draw_mask = draw_segmentation_masks(image, prediction.get('masks'), alpha=0.5) 
        
        fig = plt.figure(figsize=(15, 10))
        z = fig.add_subplot(121) 
        img = F.to_pil_image(image)
        plt.imshow(np.asarray(img))
        plt.axis('off')

        a = fig.add_subplot(122)
        width = image.shape[2]
        height = image.shape[1]
        img = F.to_pil_image(draw_mask)
        a.imshow(np.asarray(img))
        a.axis('off')
        a.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        boxes = prediction.get('boxes')
        scores = torch.round(prediction.get('scores'), decimals=1)
        for i in range(len(boxes)):
            box = boxes[i]            
            x1 = int(box[0])
            y1 = int(box[1])
            conf = scores.detach().numpy()[i]
            name = class_names[prediction.get('labels')[i]] + " " + str(conf)
            a.text(x1 , y1, name, fontsize = 12, color='white', 
                   bbox=dict(alpha=1, facecolor='black', edgecolor='black'))
    plt.show()
    

import time
from IPython.display import clear_output
from ultralytics.utils.plotting import Annotator, colors

def real_time(task='segmentation', video='webcam', id_cam=0, save=False, fig_size=(12,8), classes=None):
    try:
        if video == 'webcam':
            cap = cv2.VideoCapture(id_cam)        
        else:
            cap = cv2.VideoCapture(video)

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                               cv2.CAP_PROP_FRAME_HEIGHT, 
                                               cv2.CAP_PROP_FPS))
        if save:
            out = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
            
        if task == 'segmentation':
            model = YOLO("yolov8n-seg.pt", verbose=False)
        if task == 'detection':
            model = YOLO("yolov8n.pt", verbose=False)
        if task == 'world':
            model = YOLOWorld('yolov8s-world.pt')

        while True:
            ret, img = cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            if task == 'segmentation':
                
                names = model.model.names

                results = model.predict(img, verbose=False)
                annotator = Annotator(img, line_width=2)

                if results[0].masks is not None:
                    clss = results[0].boxes.cls.cpu().tolist()
                    masks = results[0].masks.xy
                    for mask, cls in zip(masks, clss):
                        annotator.seg_bbox(mask=mask,
                                           mask_color=colors(int(cls), True),
                                           det_label=names[int(cls)])
       

                fig = plt.figure(figsize=fig_size)
                color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(color)
                plt.axis('off')
                plt.show()
                
                clear_output(wait=True)
                
                time.sleep(0.1)


            if task == 'detection':
                results = model.predict(img, verbose=False)
                image = results[0].plot()
                fig = plt.figure(figsize=fig_size)
                color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(color)
                plt.axis('off')
                plt.show()

                clear_output(wait=True)

                time.sleep(0.1)
            
            if task == 'world':
                if classes is not None:
                    model.set_classes(classes)
                result = model.predict(img, verbose=False)
                image = result[0].plot()
                fig = plt.figure(figsize=fig_size)
                color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(color)
                plt.axis('off')
                plt.show()

                clear_output(wait=True)

                time.sleep(0.1)


            if save:
                out.write(img)

            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
                break
        if save:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        if save:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Keyboard Interrupt")
        clear_output()
        return
