# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 07:32:20 2023

@author: Kaitlin
"""

from functions.load_train_test_data import load_train_test_data
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_validate
from segmentation_models import multi_class_unet_model #Uses softmax


###############
#  Functions  #
###############

def create_model():
    # create model
    model = multi_class_unet_model(n_classes=n_classes, IMG_HEIGHT=SIZE_Y, IMG_WIDTH=SIZE_X, IMG_CHANNELS=IMG_CHANNELS)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

 
###############
#  Load data  #
###############

# new image size
SIZE_X = 512
SIZE_Y = 512

# filepaths
image_path = "../data/Master/src/"
image_type = ".jpeg"

mask_path = "../data/Master/gt/"
mask_type = ".bmp"

# model selection
#model_to_load = 'UNET_Multiclass'  #"UNET_Multiclass", "DeepLabV3Plus"

# load data (train-test-split random state set to 0 for reproducability)
X_train, y_train_cat, X_test, y_test_cat, n_classes, class_weights_dict = load_train_test_data(image_path=image_path, image_type=image_type, mask_path=mask_path, mask_type=mask_type, sizey=SIZE_Y, sizex=SIZE_X)

# define number of image channels
IMG_CHANNELS = X_train.shape[3]


##########################################
#  Tune batch size and number of epochs  #
##########################################

# create model
model = KerasClassifier(model=create_model)

# define the grid search parameters
batch_size = [8]  #[8, 16, 32, 64]
epochs = [1]  #[1, 2, 3]  #[10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
cv = cross_validate(model, X_train, y_train_cat, fit_params=param_grid, n_jobs=-1, cv=5, verbose=2)

# summarize results
#print("Best: %f using %s" % (cv_results.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))


###################################
#  Choose optimization algorithm  #
###################################




#####################################
#  Tune learning rate and momentum  #
#####################################




########################################
#  Tune network weight initialization  #
########################################
































