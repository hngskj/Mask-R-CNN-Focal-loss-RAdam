import os
import sys
import random
import math
import imutils
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import keras
ROOT_DIR = ""
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import utils
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
from samples.coco import coco

from cfg import *
from postprocessing import calc_color

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_DIR = os.path.join(ROOT_DIR, "frames2")

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
# config.display()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)


file_names = next(os.walk(IMAGE_DIR))[2]
file_names.sort()

for f in file_names:
    if f == '.DS_Store':
        continue
    image = skimage.io.imread(os.path.join(IMAGE_DIR, f))
    results = model.detect([image], verbose=1)
    r = results[0]
    
    (_, _, _, new_masks) = utils.preserve_person(r, cls_id=1)

    # visualize.display_instances(image, new_rois, new_masks, new_class_ids, class_names, new_scores)
    # visualize.write_instances('frames/i_'+f,image, new_rois, new_masks, new_class_ids, class_names, new_scores)
    
    lab_hist = calc_color.hist_lab(image, new_masks)
    # calc_color.write_lab(lab_hist, dir_name='outputs/histogram_lab/', file_name=f)
    # calc_color.plot_lab(lab_hist, dir_name='outputs/histogram_lab_plot/', file_name=f)

    # rgb_hist = calc_color.hist_rgb(image, new_masks)
    # calc_color.plot_rgb(rgb_hist, dir_name='outputs/histogram_rgb_plot/', file_name=f)

