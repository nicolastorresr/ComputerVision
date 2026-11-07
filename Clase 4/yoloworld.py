#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import time
from IPython.display import clear_output
from ultralytics.utils.plotting import Annotator, colors
import cv2
import copy
from ultralytics import YOLO, YOLOWorld
import cv2
import matplotlib.pyplot as plt

def plot_yolo_world(path, classes=None, verbose=False, fig_size=(12,8)):
    model = YOLO("yolov8s-world.pt")
    if classes is not None:
        model.set_classes(classes)
    result = model.predict(path, verbose=verbose)
    image = result[0].plot()
    fig = plt.figure(figsize=fig_size)
    color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(color)
    plt.axis('off')
    plt.show()

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

