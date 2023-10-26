# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:27:07 2023

@author: Kaitlin
"""

#############
#  Imports  #
#############

import os
import glob
import cv2
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.utils import to_categorical, normalize


###############
#  Functions  #
###############

#helper function
def get_image_data(filepath, filetype, size_x, size_y):
    image_list = []
    
    for directory_path in glob.glob(filepath):
        for img_path in glob.glob(os.path.join(directory_path, "*" + filetype)):
            img = cv2.imread(img_path, 0)  #being read as grayscale even though RGB
            img = cv2.resize(img, (size_y, size_x), interpolation=cv2.INTER_NEAREST)  # nearest neighbor interpolation to prevent creation of new mask values
            image_list.append(img)
           
    #Convert list to array for machine learning processing        
    return np.array(image_list)

#main function
def load_train_test_data(image_path, image_type, mask_path, mask_type, sizey, sizex):
   
    ############################################
    #  Load and resize train images and masks  #
    ############################################
    
    images = get_image_data(filepath=image_path, filetype=image_type, size_x=sizex, size_y=sizey)
    masks = get_image_data(filepath=mask_path, filetype=mask_type, size_x=sizex, size_y=sizey)
    
    
    ###################
    #  Encode labels  #
    ###################
    
    labelencoder = LabelEncoder()
    n, h, w = masks.shape
    masks_1_dim = masks.reshape(-1,1).ravel()
    masks_1_dim_encoded = labelencoder.fit_transform(masks_1_dim)
    masks_encoded = masks_1_dim_encoded.reshape(n, h, w)
        
    
    ###########################################
    #  Expand image dimensions and normalize  #
    ###########################################
    
    images_input = np.expand_dims(images, axis=3)
    images_input = normalize(images_input, axis=1)
    
    masks_input = np.expand_dims(masks_encoded, axis=3)
    
    
    ######################
    #  Train-test split  #
    ######################
    X_train, X_test, y_train, y_test = train_test_split(images_input, masks_input, test_size = 0.25, random_state = 0)
    
    class_values = np.unique(y_train)
    n_classes = len(class_values)
    print("Class values in the dataset are ... ", class_values)
    print("Number of classes in the dataset is ... ", n_classes)
    
    
    #################################
    #  One-hot encode mask classes  #
    #################################
    
    train_masks_cat = to_categorical(y_train, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
    
    test_masks_cat = to_categorical(y_test, num_classes=n_classes)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
    
    
    ######################################
    #  Calculate balanced class weights  #
    ######################################
    
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                     classes=np.unique(masks_1_dim_encoded),
                                                     y=masks_1_dim_encoded)
    print("Class weights are...:", class_weights)
    
    class_weights_dict = {}
    for ind, c in enumerate(np.unique(masks_1_dim_encoded)):
        class_weights_dict[c] = class_weights[c]
    print("Class weights dictionary looks like...:", class_weights_dict)
    
    return X_train, y_train_cat, X_test, y_test_cat, n_classes, class_weights_dict