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

iou_threshold = 0.5
confidence = 0.90
# Path to the images - both train and test set
# there are two sub folders here - train and val

set1 = random.choice(["train","val","test"])
images ="datasets/fruits2"
output_folder = "fruits2-20210713T1300-0192"

#Path to train and test images respectively
images_path = os.path.join(images,set1)

# Path to the annotation files - for train and test set.
# annotations = os.path.join(images,"{}/via_project_fruits.json".format(set1))
# print("Existence of {} annotation file: ".format(set1),os.path.exists(annotations))

# pick an image at random to use it as a test 
# Skipping the annotation file. Annotation file is named via_project_fruits.json
image_name = random.choice([i for i in os.listdir(images_path) if not i.startswith("via")])
filename , ext = os.path.splitext(image_name)
print("Filename: ",filename)
#path to ground truth masks - genertaed from the annotation files
# You cannot execute this before executing generate_truth-masks.py script
truth_masks  = "./evaluation/truth_masks/fruits2/{}_masks_truth".format(set1)
print("Existence of {} masks truth: ".format(set1), os.path.exists(truth_masks))


#path to prediction masks - the output of Mask R-CNN in output folder is enough for this
pred_masks  = "./output/{}/{}_masks2".format(output_folder, set1)
print("Existence of {} masks pred: ".format(set1),os.path.exists(pred_masks))


# example - just puicking one image for testing.
example_image = os.path.join(images_path,image_name)
example_truth = os.path.join(truth_masks,"{}_truth.npy".format(filename))
example_pred = os.path.join(pred_masks,"{}_mask2.npy".format(filename))

print("Example image: ", os.path.exists(example_image))
print("Example truth: ", os.path.exists(example_truth))
print("Example pred: ", os.path.exists(example_pred))



# Call MaskConstruction class 
# This class contains all functions used to reconstruct and draw masks
# Parameters: image, ground-truth masks,prediction masks and confidence
s = MaskConstruction(example_image,example_truth,example_pred,confidence)

# Draw prediction masks
# if you want to view the output pass a parameter display = True
# it is false by default
s.draw_contours(display=True)

# draw rectangular bounding boxes 
s.drawBbox(display=False)
# Call MaskEvaluation class
# This class contains all the function used to evaluate Mask-RCNN
# Parameter: IoU threshold
ss= MaskRCNN_Evaluation(iou_threshold)

# Confusion matrix
a = np.load(example_truth, allow_pickle=True)
b = np.load(example_pred, allow_pickle=True)
print([i["box"] for i in b if i["confidence"]>=confidence])
print([i["confidence"] for i in b if i["confidence"]>=confidence])

tp, fp, fn = ss.ConfusionMatrix(truth=a, preds=s.Contors())
print("True Positives", tp)
print("False Positives", fp)
print("False Negatives", fn)
# Draw ground-truth masks
# passs a parameter display = True to view the output. False by default
s.draw_truth_masks(display=True)





exit()
# Write precision and recall into a CSV file
ss.WriteAndDisplayPR(annotations,pred_masks,truth_masks,images_path,set1=set1,break_num=-1)


# Print AP for all-point interpolation method, the function also saves the AP values to evaluation/results
print(ss.AP_NoInterpolation(set1=set1)["AP"])

# Print AP for 11-point interpolation
print(ss.AP_11PointInterpolation(set1=set1)["AP"])


# Write the confusion matrix into a JSON file
ss.WriteConfusionMatrix(images_path,truth_masks,pred_masks,set1=set1)

# Plot PR curve, pass display = True parameter to display the plot. False by default
# the function also returns precision, recall, set1(train or test) and IoU at 
# different level of confidence.
precision, recall,set1, iou = ss.PlotPrecisionRecallCurve(set1=set1, display=False)


img = cv.imread(example_image)
bboxes = np.array([i["box"] for i in b])
confs = np.array([i["confidence"] for i in b])
bb = np.array([i["mask"] for i in b])

d = []
for bbox, conf in zip(bboxes, confs):
	bbox_img = img[bbox[0]:bbox[2],bbox[1]:bbox[3]]
	r = bbox_img[:,:,0].mean().astype("uint8")
	g = bbox_img[:,:,1].mean().astype("uint8")
	b = bbox_img[:,:,2].mean().astype("uint8")
	rgb = np.mean([r,g,b]).astype("uint8")
	d.append(rgb+[conf])
	print(rgb, conf)
# [378 393 442 455]
# 410 424
# bbox[:,:,2] = 255

pd.DataFrame(d).to_csv("/home/kiprono/Desktop/data.csv", index=False)




###################### END ###############################

# Everything written below is meant for testing purposes - not essentials








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