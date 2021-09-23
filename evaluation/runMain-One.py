import os
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from MaskReconstruction import MaskConstruction
from Evaluation import MaskRCNN_Evaluation
import random
import json
import pandas as pd

# changing the font size in matplotlib
import matplotlib
font = {'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

# Change the working dir into the root
os.chdir("../")


# Parameters
iou_threshold = 0.5
confidence = 0.90
kind = "random" # specific or random
output_folder = None # should be in the output folder, if left none it will default to the latest output


images_path ="assets/datasets/fruits2"

if kind=="specific":
    image_name = "_MG_9000_08.jpg" # include extension
elif kind=="random":
    image_name = random.choice(os.listdir(os.path.join(images_path, "images")))

#Path to train and test images respectively
image_set_path = [os.path.join(images_path, i) for i in ["train", "val", "test"] if os.path.exists(os.path.join(images_path, i, image_name))][0]

image_path = os.path.join(image_set_path, image_name)
assert os.path.exists(image_path),\
"Image does not exists"

filename, ext = os.path.splitext(image_name)
setq = image_set_path.split("/")[-1]

truth_masks  = "./evaluation/truth_masks/fruits2/{}_masks_truth".format(setq)

# If output folder is not specied we default into the latest output folder available.
if output_folder is None:
    assert len(os.listdir("output"))>=1,\
    "[Info] There is no output. Create the output by running fruit-detections.ipynb file in the Python folder in Root dir."

    from pathlib import Path
    output_folder = sorted(Path("output").iterdir(), key=os.path.getmtime)[-1]
#path to prediction masks - the output of Mask R-CNN in output folder is enough for this
pred_masks  = "{}/{}_masks2".format(output_folder, setq)

# example - just picking one image for testing.
truth_path = os.path.join(truth_masks,"{}_truth.npy".format(filename))
if not os.path.exists(truth_path):
    print("[Info] Truth mask not found. This could be an error or be by design. The later if there are no fruit instances on the image and former if the path is incorrect. Exitting...")
    exit()
pred_path = os.path.join(pred_masks,"{}_mask2.npy".format(filename))
assert os.path.exists(pred_path),"Predicted masks does not exist."


print("Image path:", image_path)
print("Pred path", pred_path)
print("Truth path", truth_path)

# Call MaskConstruction class 
# This class contains all functions used to reconstruct and draw masks
# Parameters: image, ground-truth masks,prediction masks and confidence
s = MaskConstruction(image_path, truth_path, pred_path, confidence)

# Draw prediction masks
# if you want to view the output pass a parameter display = True
# it is false by default
s.draw_contours(display=True)


s.drawBbox(display=False)
# Call MaskEvaluation class
# This class contains all the function used to evaluate Mask-RCNN
# Parameter: IoU threshold


# Draw ground-truth masks
# passs a parameter display = True to view the output. False by default
s.draw_truth_masks(display=True)

ss= MaskRCNN_Evaluation(save_results_to=None, iou_value=iou_threshold, confidence = confidence)

# Confusion matrix
a = np.load(truth_path, allow_pickle=True)
b = np.load(pred_path, allow_pickle=True)
print("Bounding boxes:", [list(i["box"]) for i in b if i["confidence"]>=confidence])
print("Confidence scores: ", [i["confidence"] for i in b if i["confidence"]>=confidence])

tp, fp, fn = ss.ConfusionMatrix(truth=a, preds=s.Contors())
print("True Positives", tp)
print("False Positives", fp)
print("False Negatives", fn)

