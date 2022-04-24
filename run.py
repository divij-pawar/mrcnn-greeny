import cv2
import numpy as np
import os
import sys
import coco
from mrcnn import utils
from mrcnn import model as modellib
from helpers import apply_mask, add_mask, display_instances

py_ver = sys.version_info[0:2]
if  py_ver >= (3, 9) or py_ver < (3,5) :
    raise Exception('Requires python between 3.8 and 3.5')

# Load the pre-trained model data
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
    
# Change the config infermation
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()


# COCO dataset object names
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

video_name = input("Enter video filename: ")
capture = cv2.VideoCapture(video_name)
output_file = video_name[:-4]+"_output.mp4" 

# Recording Video
fps = np.floor(capture.get(cv2.CAP_PROP_FPS))
width = int(capture.get(3))
height = int(capture.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

ct=1
while True:
    ret, frame = capture.read()
    if not ret:
        print(" Exiting ...")
        break
    results = model.detect([frame], verbose=0)
    print("Done predicting frame number ",ct)
    ct+=1
    r = results[0]
    frame = display_instances(
        frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
    )

    # Recording Video
    out.write(frame)

out.release()
capture.release()
