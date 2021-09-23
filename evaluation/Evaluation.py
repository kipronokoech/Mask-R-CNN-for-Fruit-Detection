import numpy as np
from shapely.geometry import Polygon,Point
import matplotlib.pyplot as plt
from matplotlib import rc
import shapely
import cv2 as cv
import os
import gc
import json
from MaskReconstruction import MaskConstruction
import pandas as pd
font = {'weight' : 'normal',
        'size'   : 22}
rc('font', **font)

class MaskRCNN_Evaluation(object):
    def __init__(self,save_results_to, iou_value=0.5, confidence = 0.90):
        """
        iou_value= iou treshold for TP and otherwise.
        """
        self.iou_value = iou_value
        self.confidence = confidence
        if save_results_to != None:
            self.save_results_to = os.path.join("evaluation/results", save_results_to)
            if not os.path.exists(self.save_results_to):
                os.makedirs(self.save_results_to)
            
        else:
            self.save_results_to = None

    def ConfusionMatrix(self,truth,preds):
      """
      ground= array of ground-truth contours.
      preds = array of predicted contours.
      """

      #we will use this function to check iou less than threshold
      def CheckLess(list1,val):
          num = [i for i in list1 if i>=val]
          # print(list1)
          # print(num)
          # print(len(num))
          # (all(x<=val for x in list1))
          return num

      prob1=[]
      for i in range(len(preds)):
          f1=preds[i]
          f1=shapely.geometry.Polygon(f1).buffer(0)
          f1_radius=np.sqrt((f1.area)/np.pi)
          f1_buffered=shapely.geometry.Point(f1.centroid).buffer(f1_radius*1000)
          cont=[]
          for i in range(len(truth)):
            ff=shapely.geometry.Polygon(np.squeeze(truth[i])).buffer(0)
            if f1_buffered.contains(ff)== True:
              iou=(ff.intersection(f1).area)/(ff.union(f1).area)  
           
              cont.append((iou))

          prob1.append(cont)

      fp = 0
      tp = 0
      for t in prob1:
        above_t = CheckLess(t,self.iou_value)
        # print("From Pred",t)
        # print(len(CheckLess(t,self.iou_value)))
        if not above_t:
            # print("fp with allzeros",t)
            # print(len(above_t))
            fp = fp+1
        elif len(above_t) >= 1:
            # print("tp with 1 above threshold",t)
            # print(len(above_t))
            tp = tp + 1
        # elif len(above_t)>1:
        #     print("fp for extras",t)
        #     print(len(above_t))
        #     tp = tp + 1

        #     fp = fp + (len(CheckLess(t,self.iou_value))-1)

      all_truths = truth.shape[0]
      fn = all_truths - tp
    
      # print("len(pred) ", len(preds))
      # print("fp+tp ", tp+fp)
      # print("len(truth) ", len(truth))
      # print("fn+tp ", tp+fn)
      return tp, fp, fn #, len(preds), len(truth)

    def WriteConfusionMatrix(self,images_path, truth_path,preds_path,set1,threshold=0.9):
        """
        This function write TP,FP,FN,PRECISION, RECALL into a file

        truth-path - is a path to the ground truth masks
        preds_path - is a path to the prediction masks
        """
        if os.path.exists("{}/{}ConfusionMatrix%{}%{}.json".format(self.save_results_to,set1,self.confidence, self.iou_value)):
            print("{}/{}ConfusionMatrix%{}%{}.json already exists. Quitting.".format(self.save_results_to, set1,self.confidence,self.iou_value))
            return None

        f= open("{}/{}ConfusionMatrix%{}%{}.json".format(self.save_results_to, set1,self.confidence,self.iou_value),"w+")
        all_detections = 0
        all_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_details = [] 
        for index,image_path in enumerate(os.listdir(images_path)):
            image = cv.imread(os.path.join(images_path,image_path))
            filename, ext = os.path.splitext(image_path)
            truth_name = filename + "_truth.npy"
            if not os.path.exists(os.path.join(truth_path,truth_name)):
                # if the truth mask does not exist then there is no instances thus skip
                continue
            truth = np.load(os.path.join(truth_path,truth_name),allow_pickle=True)
            # truth = np.load(os.path.join(truth_path,truth_mask), allow_pickle=True)
            pred_name = filename + "_mask2.npy"
            pred = np.load(os.path.join(preds_path,pred_name),allow_pickle=True)
            pred = [mask_ for mask_ in pred if mask_["confidence"] > self.confidence]
            s = MaskConstruction(image=image, truth=truth,mask=pred,threshold=self.confidence)
            tp, fp, fn = self.ConfusionMatrix(truth,s.Contors())
            if (tp+fp) == 0:
                continue
            precision = tp / len(pred)#(tp+fp)
            recall = tp / truth.shape[0]
            all_detections = all_detections + len(pred)#(tp+fp)
            all_gt = all_gt + truth.shape[0]
            total_tp = total_tp + tp
            total_fp = total_fp + fp
            total_fn = total_fn + fn
            r = {
                "filename": filename,
                "TP":tp,
                "FP": fp,
                "FN": fn,
                "Precision":precision,
                "Recall":recall,
                "Detections": len(pred),#(tp+fp),
                "GT":truth.shape[0]
            }
            all_details.append(r)

        summary = {
        "Total TP":total_tp,
        "Total FP": total_fp,
        "TOtal FN": total_fn,
        "All Detections": all_detections,
        "All GT": all_gt,
        "TP(%)":total_tp/all_detections,
        "FP(%)":total_fp/all_detections,
        "FN":total_fn/all_detections
        }
        all_details.insert(0,summary)
        json.dump(all_details, f,indent=3)
        f.close()
        return None

    @staticmethod
    def num_of_gts(annotations_path):
        """
        This function is meant to return the number of all ground truths for all images
        in the set (train or test)
        annotations_path - This is the path to the VGG annotations json file

        Returns : Int

        """
        with open(annotations_path,"r") as fp:
            data = fp.read()

        data = json.loads(data)
        annotations = list(data.values())
        total_gt = 0
        for index in range(len(annotations)):
            num_fruits = len(annotations[index]["regions"])
            total_gt = total_gt+num_fruits

        return total_gt


    def WriteAndDisplayPR(self,annotations_path,preds_path,truth_path,images_path,set1,break_num=10):
        """
        Purpose : Write Precision and Recall into CSV file
        annotations -  VGG annotations for all images in set (train or test)
        preds_path - path to per-image prediction mask for the set (train or test)
        truth_path - path to per-images truth mask for the set in question
        images_path - path to the images in set (train or truth)
        num - number of images to instances. It may take long so we may want to break
        out of the loop when we want to test things up.
        set1 - train or test set

        """
        assert os.path.exists(annotations_path),\
        "Annotations path is not valid. Check again."

        assert os.path.exists(images_path),\
        "Images path is not valid. Check again."

        assert os.path.exists(truth_path),\
        "Ground truth masks path is not valid. Check again."

        assert os.path.exists(preds_path),\
        "Path to the prediction masks provided is not valid. Check again."

        assert set1 in ["test", "train","val"], \
        "set3 can only be 'test', 'train' or 'val. You may have to rename yours sets to follow this naming style"

        assert break_num==-1 or break_num > 0,\
        "Choose -1 to write all annotations or any positive integer to specify the number of annotations to write"

        if os.path.exists("{}/{}AP%{}%{}.csv".format(self.save_results_to, set1,self.confidence,self.iou_value)):
            print("{}/{}AP%{}%{}.csv already exists. Quitting.".format(self.save_results_to, set1,self.confidence,self.iou_value))
            return None


        metadf = pd.DataFrame(columns=["filename","detection","confidence","TP","FP"])
        for index,image in enumerate(os.listdir(images_path)):
            if index == break_num and break_num != -1:
                break
            try:
                #Load image
                image_ = os.path.join(images_path,image) 
                filename, ext = os.path.splitext(image)
                # Load prediction mask
                pred = os.path.join(preds_path,filename+"_mask2.npy")
                pred = np.load(pred, allow_pickle = True)
                # Load truth mask
                truth = os.path.join(truth_path,filename+"_truth.npy")
                truth = np.load(truth, allow_pickle = True)
                # Determine the number of detections per image
                num  = len(pred)
                for i in range(num):
                    #Call MaskConstruction
                    s = MaskConstruction(image_,truth,[pred[i]],0.0)
                    #print(pred[i])
                    # generate contours for a given detection
                    cont = np.array(s.Contors())
                    tp,fp, _ = self.ConfusionMatrix(truth, cont)
                    # Take care of multiple detections overlapping one gt
                    if tp>1:
                        tp = 1
                        fp = fp + (tp-1)
                    #load a detection mask with the confidence
                    mask,confidence = s.load_mask_and_confidence()
                    # create a dataframe for details of one detection
                    df = pd.DataFrame([[filename,i,confidence,tp,fp]],\
                        columns=["filename","detection","confidence","TP","FP"])
                    # concatenate the df of one detection to a mega one
                    metadf = pd.concat([metadf,df],axis=0,ignore_index=True)
            except FileNotFoundError as s:
                pass
                # escape annotation file and images with no fruits under truth annotations
                # print(s)
                # print("Skipped the annotation file or no predicted instances")

        # sort the pandas DF based on confidence
        metadf = metadf.sort_values(by="confidence",ascending=False)
        # Calculate cumulative for True Positive and False Negative
        metadf['TP-CUM'] = pd.Series(metadf["TP"].cumsum(), index=metadf.index)
        metadf['FP-CUM'] = pd.Series(metadf["FP"].cumsum(), index=metadf.index)
        metadf["all_detections"] = metadf["TP-CUM"] + metadf["FP-CUM"]
        metadf["Precision"] = metadf["TP-CUM"]/metadf["all_detections"]
        all_gt = self.num_of_gts(annotations_path)
        metadf["Recall"] = metadf["TP-CUM"]/all_gt

        #save the DF as csv file
        metadf.to_csv("{}/{}AP%{}%{}.csv".format(self.save_results_to, set1,self.confidence,self.iou_value), index = False)

        return metadf


    def PlotPrecisionRecallCurve(self,set1,display=False):
        assert os.path.isfile("{}/{}AP%{}%{}.csv".format(self.save_results_to, set1,self.confidence,self.iou_value)),\
        "File containing precision and recall for this set does not exist. Please\
        run WriteAndDisplayPR() function"
        df = pd.read_csv("{}/{}AP%{}%{}.csv".format(self.save_results_to, set1,self.confidence,self.iou_value))
        precision = df["Precision"]
        recall = df["Recall"]
        if os.path.exists("{}/PrecisionRecallCurve_{}AP%{}%{}.png".format(self.save_results_to, set1,self.confidence,self.iou_value)):
            print("File already exists >>","{}/PrecisionRecallCurve_{}AP%{}%{}.png".format(self.save_results_to, set1,self.confidence,self.iou_value))
            if display == True:
                pr_plot = plt.imread("{}/PrecisionRecallCurve_{}AP%{}%{}.png".format(self.save_results_to, set1,self.confidence,self.iou_value))
                plt.imshow(pr_plot)
                plt.show()
            return precision, recall, set1, self.iou_value
        plt.figure(figsize=(12,10))
        plt.grid(True)
        plt.plot(recall,precision,label="{}@{}".format(set1,self.iou_value),linewidth=2)
        #plt.plot(precision75,recall75,label="train@0.75")
        plt.title("Precision x Recall curve\n")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.savefig("{}/PrecisionRecallCurve_{}AP%{}%{}.png".format(self.save_results_to, set1,self.confidence,self.iou_value))
        print("The plot has been saved as: PrecisionRecallCurve_{}".format("{}AP{}%{}%.png".format(set1,self.confidence,self.iou_value)))
        if display == True:
        	plt.show()
        return precision, recall, set1, self.iou_value


    def AP_NoInterpolation(self,set1):

        assert os.path.isfile("{}/{}AP%{}%{}.csv".format(self.save_results_to, set1,self.confidence,self.iou_value)),\
        "File containing precision and recall for this set does not exist. Please\
        run WriteAndDisplayPR() function"

        df = pd.read_csv("{}/{}AP%{}%{}.csv".format(self.save_results_to, set1,self.confidence,self.iou_value))

        precision = df["Precision"]
        recall = df["Recall"]

        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in recall]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in precision]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]

        r = {
        "AP":ap,
        "Interpolated Precision": mpre,
        "Interpolated Recall": mrec,
        "Misc" : ii
        }
        with open("{}/{}AP_valuesAP_NoInterpolation%{}%{}.json".format(self.save_results_to, set1,self.confidence,self.iou_value),"w+") as fp:
            json.dump(r,fp,indent=3)
        return r


    def AP_11PointInterpolation(self,set1):

        assert os.path.isfile("{}/{}AP%{}%{}.csv".format(self.save_results_to, set1,self.confidence,self.iou_value)),\
        "File contating precision and recall for this set does not exist. Please\
        run WriteAndDisplayPR() function"

        df = pd.read_csv("{}/{}AP%{}%{}.csv".format(self.save_results_to, set1,self.confidence,self.iou_value))
        precision = df["Precision"]
        recall = df["Recall"]

        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in recall]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in precision]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]

        r = {
        "AP" : ap,
        "rhoInterp": rhoInterp,
        "recallValues" : recallValues,
        "Misc": None
        }

        with open("{}/{}AP_values_11_PointInterpolation%{}%{}.json".format(self.save_results_to, set1,self.confidence,self.iou_value),"w+") as fp:
            json.dump(r,fp,indent=3)

        return r