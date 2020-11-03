# Mask R-CNN for Fruit Detection in an Orchard 
## Dataset
The dataset used in this project were collected from 3 sources namely:
- Aerobotics dataset - The images are sourced by flying the drone ∼ 2m above the tree canopy which generates 2.7k video imagery. These short videos are then segmented to generate images. 
- FUJI dataset [link](https://zenodo.org/record/3715991)
- ACFR dataset [link](http://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/)
## Setup
- Python virtual environment - Link this to my Medium article.
- Upgrade pip and setup tools
- Requirements on requirement.txt
- Link to the dataset and anotations

## Directory Structure
Repository: Mask-R-CNN-for-Fruit-Detection
```bash
Root:
├── Python
├── assets├── datasets├── fruits├── train > (train images and annotation file.)
│         │                     ├── val > (test images and annotation file.)
│         ├── history > training-stats
│         ├── logs > trained-models
├── evaluation
├── mrcnn├── __init__.py
│        ├── config.py   
│        ├── model.py
│        ├── parallel_model.py
│        ├── utils.py
│        ├── visualize.py
├── setup.py
├── README.md
├── requirements.txt
├── via.html
└── .gitignore
```

## Detailed description of repository content
- `Python` - This folder contains `fruits.ipynb` notebook. Mask R-CNN model is trained and tested in this notebook.
- `assets` - This folder contains 3 sub-directories datasets, history, and logs:
	- datasets/fruits/train - this folder consist of training images and corresponding JSON annotations file.
	- datasets/fruits/val - contains testing images and the JSON file with the annotations.
	- history - this directory holds (will hold) the training statistics - accuracy and losses.
	- logs - trained model is saved here. For any particular model training instance a subdirectory will be created and model saved at each epoch. The created directory will be named in this format: {class_name}{date}T{time}, for example, the reposity contains  fruit20200802T0017 for the model training that was initiated on Aug,2 2020 at 0017. 
- `evaluation` - Trained model is evaluated using files in this directory. The folder contains the following dirs, subdirs and files:
	- `metrics.pdf` - This PDF files discusses the following: The original source of data (3 sources), the metrics used to evaluate the model and the perfomance of Mask R-CNN on fruit detection task based on those metrics.
	- `results` - contains all the results for the metric used to evaluate the model - Confusion Matrix, Precision , Recall, Average precision and Precision x Recall curve.
	- `generate_truth_masks.py` - This script is used to generate the annotations/labels for each image. This is important for the purposes of per-image evaluation.
	(Ideally, this should be the first script to be executed in the process of evaluation). Executing this script creates `truth_masks` folder which contain per-image ground-truth masks for both train and test set. 
	- `Evaluation.py` contains a class that defines all the metrics used in the project: Confusion matrix, AP, Precision and Recall.

	- `MaskReconstruction.py` - This script contains all functions related to manipulation of model output from contour reconstruction to drawing and writing contors.
	- `runMain.py` - Running this script calls MaskRCNN_Evaluation class in Evaluation.py. The script is mainly used to generate and save the results (important).
- `mrcnn` - this folder contains all the core files needed to train Mask R-CNN. The model itself is defined in `model.py`. Other files in the folder includes `config.py` (contains Configuration class for Mask R-CNN), `parallel_model.py` (to set up parallel processing), `utils.py` (contains common utility functions and classes), `visualize.py` (facilitate visualization of model output).
- `requirements.txt` - contains all the libraries and packages required run the model. Specific versions of libraries are defined to ease reproducibility.
- `setup.py` - This file is executed as a part of setup process. The process installs the necessary dependencies that are missing. Once you have gone through `Setup` section executing this file won't be necessary.
- `via.html` - This is fully-fledged VGG annotator. The online version of the annotator can be accessed here.
[here.](http://www.robots.ox.ac.uk/~vgg/software/via/via.html)

## Sample images in the dataset
Here are some samples of the images used in this project
[images] 
## Training progress plot

## Sample Mask RCNN results

## Evaluation
 - Sample of Mask RCNN with segmentation masks extracted
 - Brief description of the metrics: precision, recall, F1, AP, and PR curve
 - Tabulate the perfomance of the model and plot the curves where applicable.
 - Link to the paper work containing the details about the results. Include the paper on the repo.