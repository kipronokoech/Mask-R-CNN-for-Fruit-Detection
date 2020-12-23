import os
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from MaskReconstruction import MaskConstruction
from Evaluation import MaskRCNN_Evaluation
import random
import json

# changing the font size in matplotlib
import matplotlib
font = {'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

# Change the working dir into the root
os.chdir("../")

iou_threshold = 0.2
confidence = 0.95
# Path to the images - both train and test set
# there are two sub folders here - train and val
images ="assets/datasets/fruits"

#Path to train and test images respectively
images_path_train = os.path.join(images,"train")
images_path_test = os.path.join(images,"val")

# Path to the annotation files - for train and test set.
train_annotations = os.path.join(images,"train/via_project_fruits.json")
test_annotations = os.path.join(images,"val/via_project_fruits.json")

# pick an image at random to use it as a test 
# Skipping the annotation file. Annotation file is named via_project_fruits.json
image_name = random.choice([i for i in os.listdir(images_path_test) if not i.startswith("via")])
filename , ext = os.path.splitext(image_name)

#path to ground truth masks - genertaed from the annotation files
# You cannot execute this before executing generate_truth-masks.py script
train_masks_truth  = "./evaluation/truth_masks/train_masks_truth"
test_masks_truth = "./evaluation/truth_masks/test_masks_truth"

#path to prediction masks - the output of Mask R-CNN in output folder is enough for this
train_masks_pred  = "./output/train_masks_pred"
test_masks_pred = "./output/test_masks_pred"


# example - just puicking one image for testing the code.
example_image = os.path.join(images_path_test,image_name)
example_truth = os.path.join(test_masks_truth,"{}_truth.npy".format(filename))
example_pred = os.path.join(test_masks_pred,"{}_mask2.npy".format(filename))

# Call MaskConstruction class 
# This class contains all functions used to reconstruct and draw masks
# Parameters: image, ground-truth masks,prediction masks and confidence

#--------------------------------------------------------------------------
# example_image = "/home/kiprono/Desktop/_MG_7954_03.jpg"
# example_pred = "/home/kiprono/Desktop/_MG_7954_03_mask2.npy"
# example_truth = "/home/kiprono/Desktop/_MG_7954_03_truth.npy"

#--------------------------------------------------------------------------
s = MaskConstruction(example_image,example_truth,example_pred,confidence)

# Draw prediction masks
# if you want to view the output pass a parameter display = True
# it is false by default
img_contors = s.draw_contours(display=False)

# plt.imshow(img_contors)
# plt.savefig("/home/kiprono/Desktop/img_contors%90.png")

# Call MaskEvaluation class
# This class contains all the function used to evaluate Mask-RCNN
# Parameter: IoU threshold
ss= MaskRCNN_Evaluation(iou_threshold,confidence)

# Draw ground-truth masks
# passs a parameter display = True to view the output. False by default
img_truth = s.draw_truth_masks(display=False)
# plt.imshow(img_truth)
# plt.savefig("/home/kiprono/Desktop/img_truth_contors.png")

# Write precision and recall into a CSV file
ss.WriteAndDisplayPR(train_annotations,train_masks_pred,train_masks_truth,images_path_train,set1="train",break_num=-1)
ss.WriteAndDisplayPR(test_annotations,test_masks_pred,test_masks_truth,images_path_test,set1="test",break_num=-1)

# Print AP for all-point interpolation method, the function also saves the AP values to evaluation/results
print(ss.AP_NoInterpolation(set1="train")["AP"])
print(ss.AP_NoInterpolation(set1="test")["AP"])

# Print AP for 11-point interpolation
print(ss.AP_11PointInterpolation(set1="train")["AP"])
print(ss.AP_11PointInterpolation(set1="test")["AP"])

# Write the confusion matrix into a JSON file
ss.WriteConfusionMatrix(test_masks_truth,test_masks_pred,set1="test")
ss.WriteConfusionMatrix(train_masks_truth,train_masks_pred,set1="train")

# Plot PR curve, pass display = True parameter to display the plot. False by default
# the function also returns precision, recall, set1(train or test) and IoU at 
# different level of confidence.
precision, recall,set1, iou = ss.PlotPrecisionRecallCurve(set1="train",display=False)
precision, recall,set1, iou = ss.PlotPrecisionRecallCurve(set1="test",display=False)




###################### END ###############################

# Everything written below is meant for testing ideas - not essentials








# data2 = {
# 			"set": set1,
# 			"iou":iou,
# 			"precision": list(precision), 
# 			"recall": list(recall)
# 		}
# if not os.path.exists("./evaluation/results/auc/{}{}pr.json".format(set1,iou)):
# 	with open("./evaluation/results/auc/{}{}pr.json".format(set1,iou),"w+") as outfile:
# 		json.dump(data2, outfile, indent = 3)
# else:
# 	print("./evaluation/results/auc/{}{}pr.json already exists".format(set1,iou))


# precision, recall,set1, iou = ss.PlotPrecisionRecallCurve(set1="test")
# data1 = {
# 		"set": set1,
# 		"iou":iou,
# 		"precision": list(precision), 
# 		"recall": list(recall)
# 	}
# set1, iou = "test", iou_threshold
# if not os.path.exists("./evaluation/results/auc/{}{}pr.json".format(set1,iou)):
# 	with open("./evaluation/results/auc/{}{}pr.json".format(set1,iou),"w+") as outfile:
# 		json.dump(data1, outfile, indent = 3) 
# else:
# 	print("./evaluation/results/auc/{}{}pr.json already exists".format(set1,iou))