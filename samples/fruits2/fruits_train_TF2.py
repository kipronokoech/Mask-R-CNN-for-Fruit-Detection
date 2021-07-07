#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive,files
drive.mount("/content/gdrive", force_remount = True)


# In[2]:


# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials

# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)


# In[2]:


import os
os.chdir('/content/gdrive/My Drive/2/Mask_RCNN_TF2')


# In[3]:


# !pip install --quiet --upgrade -r requirements.txt


# In[4]:


import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf
print(tf.__version__)
# import keras
# print(keras.__version__)
import random
import matplotlib.pyplot as plt


# In[5]:


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[6]:


import sys
sys.path.append("./mrcnn")  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# In[7]:


ROOT_DIR = './'
DATA_DIR = './datasets/fruits2'
DEFAULT_LOGS_DIR = './assets/logs'


# In[8]:


len(os.listdir(os.path.join(DATA_DIR, "train-images")))


# In[9]:


# Local path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_WEIGHTS_PATH):
    utils.download_trained_weights(COCO_WEIGHTS_PATH)
else:
    print("COCO weights already exists")


# In[10]:


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fruits2-"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU =   1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200

    # Skip detections with the following confidence level
    DETECTION_MIN_CONFIDENCE = 0.90

    # Initial weights
    # INIT_IT = "imagenet"
    INIT_IT = "last"


# In[11]:


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("balloon", 1, "balloon")

        # Train or validation dataset?
        assert subset in ["train-images", "val-images"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }

        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_project_fruits.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.float32) #dtype=np.int32

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)       


# In[12]:


# Training dataset
dataset_train = BalloonDataset()
dataset_train.load_balloon(DATA_DIR, "train-images")
dataset_train.prepare()

# Validation dataset
dataset_val = BalloonDataset()
dataset_val.load_balloon(DATA_DIR, "val-images")
dataset_val.prepare()


# In[ ]:


# Create model in training mode
# model = modellib.MaskRCNN(mode="training", config=opt,
#                           model_dir=opt.MODEL_DIR)
config = BalloonConfig()

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=DEFAULT_LOGS_DIR)

# Which weights to start with?
init_with = config.INIT_IT  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    if not os.path.exists(opt.COCO_MODEL_PATH):
        utils.download_trained_weights(opt.COCO_MODEL_PATH)
    
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(opt.COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    last_weight_path =     os.path.join(os.getcwd(),"assets/logs/fruits2-20210627T0034/mask_rcnn_fruits2-_0040.h5")
    # Load the last model you trained and continue training
    model.load_weights(last_weight_path, by_name=True)
    

'''
train
1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers 
    (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, 
    pass layers='heads' to the train() function.
2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. 
    Simply pass layers="all to train all layers.
'''

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

# Fine tuning "all" layers

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=400, 
            layers='all')

# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=80,
#             layers='all')

# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE/10,
#             epochs=120,
#             layers='all')
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE/20,
#             epochs=200,
#             layers='all')
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE/30,
#             epochs=280,
#             layers='all')
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE/30,
#             epochs=360,
#             layers='all')


# In[ ]:




