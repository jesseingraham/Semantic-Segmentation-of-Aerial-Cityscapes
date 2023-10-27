# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:46:34 2023

@author: Kaitlin

#Hyperparameter GridsearchCV
epochs = []
batch_size = []

optimization_algorithm = []

learning_rate = []
momentum = []

network_weight_initialization = []

activation_function = []

dropout_regularization = []





GridsearchCV is failing to allocate memory. Maybe try manually grid searching with just a CV function and record results manually.
1. Trying to change njobs from -1 to 1
    a. Sklearn expects X and y to not be more than 2dims. This is incompatible with the NN that expects None,512,512,1 dims.
        - proposed solutions: (1) add function to beginning and end of model that reshapes the inputs/outputs to match dims to allow passage through sklearn. (might work)
                              (2) manually do GridsearchCV with for loops (will for sure work)


CONTINUING WITH SOLUTION (2)





"""


from functions.load_train_test_data import load_train_test_data
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
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
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5, verbose=2)
grid_result = grid.fit(X_train, y_train_cat)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


###################################
#  Choose optimization algorithm  #
###################################




#####################################
#  Tune learning rate and momentum  #
#####################################




########################################
#  Tune network weight initialization  #
########################################
































