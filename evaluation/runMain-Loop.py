import os
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from MaskReconstruction import MaskConstruction
from Evaluation import MaskRCNN_Evaluation
import random
import json

os.chdir("../")

# changing the font size in matplotlib
import matplotlib
font = {'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

# Change the working dir into the root

# iou_threshold = 0.5
confidence_value = 0.90

setq = "fruits220210922T0742-0071"
set2 = "fruits2"

iou_thresholds = [0.2, 0.3, 0.4, 0.5]
sets = ["train","val", "test"]
for iou_threshold in iou_thresholds:
    for set3 in sets:
        # set2 = "fruits2"
        # set3 = "train-images"
        # Path to the images - train,val test set

        # there are two sub folders here - train and val
        images ="assets/datasets/{}".format(set2)

        #Path to train and test images respectively
        images_path = os.path.join(images,set3)

        #path to ground truth masks - genertaed from the annotation files
        # You cannot execute this before executing generate_truth-masks.py script
        truth_masks  = "./evaluation/truth_masks/{}/{}_masks_truth".format(set2,set3)
        # print(os.path.exists(truth_masks))

        #path to prediction masks - the output of Mask R-CNN in output folder is enough for this
        pred_masks  = "./output/{}/{}_masks2".format(setq,set3)
        # Path to the annotation files - for train and test set.
        set_annotations = os.path.join(images,"{}/via_project_{}.json".format(set3,set2.replace("-","_")))
        # print("{}/via_project_{}.json".format(set3,set2)) #.replace("-","_")
        # print(os.path.exists(set_annotations))

        # pick an image at random to use it as a test 
        # Skipping the annotation file. Annotation file is named via_project_fruits.json
        image_name = random.choice([i for i in os.listdir(images_path) if not i.startswith("via")])
        filename , ext = os.path.splitext(image_name)

        try:
                # example - just puicking one image for testing.
                example_image = os.path.join(images_path,image_name)
                example_truth = os.path.join(truth_masks,"{}_truth.npy".format(filename))
                example_pred = os.path.join(pred_masks,"{}_mask2.npy".format(filename))

                # print("Example Image",os.path.exists(example_image))
                # print(os.path.exists(example_truth))
                # print(os.path.exists(example_pred))

                # Call MaskConstruction class 
                # This class contains all functions used to reconstruct and draw masks
                # Parameters: image, ground-truth masks,prediction masks and confidence
                s = MaskConstruction(example_image,example_truth,example_pred,0.9)

                # Draw prediction masks
                # if you want to view the output pass a parameter display = True
                # it is false by default
                s.draw_contours(display=False)
                
                # Draw ground-truth masks
                # passs a parameter display = True to view the output. False by default
                s.draw_truth_masks(display=False)
        except:
                print("Image not located or no predicted instances. Passing")
                pass
        # Call MaskEvaluation class
        # This class contains all the function used to evaluate Mask-RCNN
        # Parameter: IoU threshold
        ss= MaskRCNN_Evaluation(iou_value=iou_threshold, confidence = confidence_value, save_results_to=setq)

        # Write precision and recall into a CSV file
        ss.WriteAndDisplayPR(set_annotations,pred_masks,truth_masks,images_path,set1=set3,break_num=-1)



        # Print AP for all-point interpolation method, the function also saves the AP values to evaluation/results
        print(ss.AP_NoInterpolation(set1=set3)["AP"])

        # Print AP for 11-point interpolation
        print(ss.AP_11PointInterpolation(set1=set3)["AP"])

        # Write the confusion matrix into a JSON file
        ss.WriteConfusionMatrix(images_path, truth_masks, pred_masks, set1=set3)

        # Plot PR curve, pass display = True parameter to display the plot. False by default
        # the function also returns precision, recall, set2(train or test) and IoU at 

        precision, recall,set1, iou = ss.PlotPrecisionRecallCurve(set1=set3, display=False)

        ###################### END ###############################

        # Everything written below is meant for testing purposes - not essentials








        # data2 = {
        #                       "set": set2,
        #                       "iou":iou,
        #                       "precision": list(precision), 
        #                       "recall": list(recall)
        #               }
        # if not os.path.exists("./evaluation/results/auc/{}{}pr.json".format(set2,iou)):
        #       with open("./evaluation/results/auc/{}{}pr.json".format(set2,iou),"w+") as outfile:
        #               json.dump(data2, outfile, indent = 3)
        # else:
        #       print("./evaluation/results/auc/{}{}pr.json already exists".format(set2,iou))


        # precision, recall,set2, iou = ss.PlotPrecisionRecallCurve(set2="test")
        # data1 = {
        #               "set": set2,
        #               "iou":iou,
        #               "precision": list(precision), 
        #               "recall": list(recall)
        #       }
        # set2, iou = "test", iou_threshold
        # if not os.path.exists("./evaluation/results/auc/{}{}pr.json".format(set2,iou)):
        #       with open("./evaluation/results/auc/{}{}pr.json".format(set2,iou),"w+") as outfile:
        #               json.dump(data1, outfile, indent = 3) 
        # else:
        #       print("./evaluation/results/auc/{}{}pr.json already exists".format(set2,iou))