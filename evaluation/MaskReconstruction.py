import numpy as np
import cv2 as cv
import json
import os
from matplotlib import pyplot as plt
import random
import ntpath

class MaskConstruction(object):
	def __init__(self,image,truth=None,mask=None,threshold = 0.9):
		self.image = image
		self.truth = truth
		self.mask = mask
		self.threshold = threshold

	def load_image(self):
		"""
		This function just read the image and returns it.
		"""
		image = cv.imread(self.image)
		image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
		return image

	def load_truth_mask(self):
		"""
		This function loads the truth masks (annotations)
		"""
		truth = np.load(self.truth,allow_pickle=True)
		return truth

	def load_pred_mask(self):
		"""
		Load prediction masks
		"""
		pred = np.load(self.mask,allow_pickle=True)
		return pred

	def only_probs(self):
		#try loading the array
		try: 
			mask1 = self.load_pred_mask()
		# else assume that the array as been loaded already
		except:
			mask1 = self.mask
		try:
			only_mask = [mask1[index]["mask"] for index in range(len(mask1))\
			if mask1[index]["confidence"]>self.threshold]
			return np.array(only_mask)

		except TypeError as e:
			print(e)

	def load_mask_and_confidence(self):
		return self.mask[0]["mask"],self.mask[0]["confidence"]

	def findContors(self,binary_img):
		#Filling up holes
		kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
		img = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel)

		# FINDING CONTOURS using findContours on OpenCV.
		thresh, im_bw = \
		cv.threshold(np.uint8(img), 127, 255, cv.THRESH_BINARY) #im_bw: binary image
		contours, hierarchy = \
		cv.findContours(im_bw,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		pts = []
		n = len(contours[0])
		for i in range(n):
			xx = contours[0][i][0][0]
			yy = contours[0][i][0][1]
			pts.append([xx,yy])
			if i == (n-1):
				x1 = contours[0][0][0][0]
				y1 = contours[0][0][0][1]
				pts.append([x1,y1])
		return pts

	def Contors(self):
		mask = self.only_probs()
		np.putmask(mask,mask>=0.5,255)
		all_masks = []
		for index in range(mask.shape[0]):
			cont = self.findContors(mask[index])
			all_masks.append(cont)
		return all_masks

	def draw_contours(self,save_to=None, display=False):
		contours = self.Contors()
		n = len(contours)
		image = self.load_image()
		for j in range(n):
			ctr = contours[j]
			img = cv.polylines(image,[np.int32(ctr)],
				isClosed=True,color=(255,0,0),thickness=5)
		if save_to != None:
			filename, ext = os.path.splitext(ntpath.basename(self.image))
			pred_name = filename+"_predmask.png"
			img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
			cv.imwrite(os.path.join(save_to,pred_name),img)
		if display == True:
			plt.imshow(img)
			plt.show()
		return img
		
	def draw_truth_masks(self,save_to = None, display = False):
		truth = self.load_truth_mask()
		image = self.load_image()
		for index in range(len(truth)):
				img = cv.polylines(image,[np.int32(truth[index])],
								isClosed=True,color=(0,0,255),thickness=5)
		if save_to != None:
			filename, ext = os.path.splitext(ntpath.basename(self.image))
			truth_name = filename+"_truthmask.png"
			img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
			cv.imwrite(os.path.join(save_to,truth_name),img)
		if display==True:
			plt.imshow(img)
			plt.show()
		return img
