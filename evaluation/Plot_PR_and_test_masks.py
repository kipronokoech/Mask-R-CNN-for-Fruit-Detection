import json
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from MaskReconstruction import MaskConstruction
from Evaluation import MaskRCNN_Evaluation
import os
import random
import matplotlib
font = {'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)
os.chdir("../")

#path to the images
images ="assets/datasets/fruits"
test_images = os.path.join(images,"val")
#path to prediction masks
preds = "output/test_masks_pred"
#path to truth masks
truths = "evaluation/truth_masks/test_masks_truth"

image = random.choice([i for i in os.listdir(test_images) if not i.startswith("via")])
# image = "_MG_8080_10.jpg"
image_path = os.path.join(test_images,image)

filename, ext = os.path.splitext(image)
pred = filename+"_mask2.npy"

pred_path = os.path.join(preds,pred)

truth = filename+"_truth.npy"

truth_path = os.path.join(truths,truth)

print(os.path.exists(image_path))
print(os.path.exists(pred_path))
print(os.path.exists(truth_path))
s = MaskConstruction(image= image_path,truth= truth_path,mask = pred_path,threshold=0.9)
s.draw_contours(display=True)
# truth = np.load(truth_path)
# image = cv.imread(image_path)
# for index in range(len(truth)):
# 		img = cv.polylines(image,[np.int32(truth[index])],
# 						isClosed=True,color=(0,0,255),thickness=7)
# plt.imshow(img)
# plt.show()

# s.draw_truth_masks()

# examples=["20151124T030346.028628_i1852j1727.png","20151124T030923.095249_i1840j641.png",\
#           "_MG_8080_10.jpg","20151124T045040.865259_i2077j1170.png","20130320T013314.338742_41.png",\
#           "_MG_7954_09.jpg","20130320T005545.911690.Cam6_41.png"]

# examples = ["20130320T005827.438900.Cam6_53.png"]
# #examples = ["_MG_8993_19.jpg","_MG_3168_08.jpg","20130320T013330.148570_32.png","20130320T005827.438900.Cam6_53.png"]

# desktop = "/home/kiprono/Desktop"
# for image in examples:
# 	image_path = os.path.join(test_images,image)

# 	filename, ext = os.path.splitext(image)
# 	pred = filename+"_mask2.npy"
# 	pred_path = os.path.join(preds,pred)
	
# 	truth = filename+"_truth.npy"
# 	truth_path = os.path.join(truths,truth)

# 	s = MaskConstruction(image_path,truth_path,pred_path,0.9)
# 	#remove continue if you want to draw contours
# 	pred = s.Contors()

# 	truth = np.load(truth_path)

# 	ss =  MaskRCNN_Evaluation(0.3)
# 	tp, fp, fn = ss.ConfusionMatrix(truth=truth, preds=pred)
# 	# print("TP",tp,"FP",fp, "FN",fn,"Preds",pred,"Truth",truth)
# 	# print(tp+fp==pred, tp+fn==truth)
# 	s.draw_contours()
# 	s.draw_truth_masks()

#Plot PR cure for training data
set1 = "train"
with open("./evaluation/results/auc/{}0.2pr.json".format(set1)) as infile:
	data02 = json.load(infile)
with open("./evaluation/results/auc/{}0.3pr.json".format(set1)) as infile:
	data03 = json.load(infile)
with open("./evaluation/results/auc/{}0.4pr.json".format(set1)) as infile:
	data04 = json.load(infile)
with open("./evaluation/results/auc/{}0.5pr.json".format(set1)) as infile:
	data05 = json.load(infile)

plt.figure(figsize=(12,10))
plt.grid(True)
plt.title("Precision x Recall curves for {} set\n".format(set1))
plt.plot(data02["recall"],data02["precision"],label="IoU@0.2",linewidth=2)
plt.plot(data03["recall"],data03["precision"],label="IoU@0.3",linewidth=2)
plt.plot(data04["recall"],data04["precision"],label="IoU@0.4",linewidth=2)
plt.plot(data05["recall"],data05["precision"],label="IoU@0.5",linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.savefig("./evaluation/results/auc/PRCurve_forDifferentThresholds_{}AP.png".format(set1))
plt.show()

#Plot PR curve for testing data
set1 = "test"
with open("./evaluation/results/auc/{}0.2pr.json".format(set1)) as infile:
	data02 = json.load(infile)
with open("./evaluation/results/auc/{}0.3pr.json".format(set1)) as infile:
	data03 = json.load(infile)
with open("./evaluation/results/auc/{}0.4pr.json".format(set1)) as infile:
	data04 = json.load(infile)
with open("./evaluation/results/auc/{}0.5pr.json".format(set1)) as infile:
	data05 = json.load(infile)

plt.figure(figsize=(12,10))
plt.grid(True)
plt.title("Precision x Recall curves for {} set\n".format(set1))
plt.plot(data02["recall"],data02["precision"],label="IoU@0.2".format(set1),linewidth=2)
plt.plot(data03["recall"],data03["precision"],label="IoU@0.3".format(set1),linewidth=2)
plt.plot(data04["recall"],data04["precision"],label="IoU@0.4".format(set1),linewidth=2)
plt.plot(data05["recall"],data05["precision"],label="IoU@0.5".format(set1),linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.savefig("./evaluation/results/auc/PRCurve_forDifferentThresholds_{}AP.png".format(set1))
plt.show()