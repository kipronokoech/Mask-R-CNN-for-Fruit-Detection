import os
import json
import numpy as np
os.chdir("../")

if not os.path.exists("./evaluation/truth_masks"):
	os.mkdir("./evaluation/truth_masks")

sets = ["train","val"]
#Looping throught both test and train set.
for set_ in sets:
	#Define the annotations file
	# if the set is train then the file is in "train" folder
	# else if the set is test the file is in the "val" folder
	file_ = "assets/datasets/fruits/{}/via_project_fruits.json".format(set_)

	#Load the fole content - note that it is loaded as string.
	with open(file_,"r") as fp:
		data = fp.read()
	#Convert the content into dictionary by calling JSON loads
	data = json.loads(data)
	# We are interested in the values of the JSON file
	annotations = list(data.values())

	# loop through all the annotations in all the images
	for t in range(len(annotations)):
		# print(annotations[t]["filename"])
		filename, ext = os.path.splitext(annotations[t]["filename"])
		# print(filename)
		all_masks = []
		for i in range(len(annotations[t]["regions"])):
			xx = annotations[t]["regions"][i]["shape_attributes"]["all_points_x"]
			yy = annotations[t]["regions"][i]["shape_attributes"]["all_points_y"]
			one = []
			for j,k in zip(xx,yy):
				one.append([j,k])
			all_masks.append(one)
		new_name = filename+"_truth.npy"

		if set_ == "val":
			set_name = "test"
		else:
			set_name = "train"

		if not os.path.exists("evaluation/truth_masks/{}_masks_truth".format(set_name)):
			os.mkdir("evaluation/truth_masks/{}_masks_truth".format(set_name))
		to_path = os.path.join(os.getcwd(),"evaluation/truth_masks/{}_masks_truth".format(set_name),new_name)
		if os.path.exists(os.path.join(os.getcwd(),"evaluation/truth_masks/{}_masks_truth".format(set_name),new_name)):
			continue
		# save the truth masks as numpy array
		np.save(to_path, np.array(all_masks))



