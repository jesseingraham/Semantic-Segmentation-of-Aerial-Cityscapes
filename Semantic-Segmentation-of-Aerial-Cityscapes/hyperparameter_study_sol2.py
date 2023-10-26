# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:20:30 2023

@author: Kaitlin

100 epochs, batch size 8, 5 fold split took 7.5 hours

"""


#############
#  Imports  #
#############

from functions.load_train_test_data import load_train_test_data
from segmentation_models import multi_class_unet_model #Uses softmax
from keras.callbacks import CSVLogger
import os


###############
#  Load data  #
###############

#new image size
SIZE_X = 512
SIZE_Y = 512

#filepaths
image_path = "../data/Master/src/"
image_type = ".jpeg"

mask_path = "../data/Master/gt/"
mask_type = ".bmp"

#load data
X_train, y_train_cat, X_test, y_test_cat, n_classes, class_weights_dict = load_train_test_data(image_path=image_path, image_type=image_type, mask_path=mask_path, mask_type=mask_type, sizey=SIZE_Y, sizex=SIZE_X)


######################
#  Cross-validation  #
######################

folds = 5  #cross-validation folds

def get_fold_indices(n_samples, n_folds):
    x = n_samples // n_folds
    
    indices_by_fold = {}
    
    for n in range(n_folds):
        fold_n = [i for i in range(x*n, x*n+x)]
        
        if n == (n_folds-1):
            fold_n.extend([i for i in range(max(fold_n)+1,n_samples)])
        
        indices_by_fold[n] = fold_n
    
    return indices_by_fold


def get_train_val_fold_split(indices_by_fold, current_fold):
    """
    Parameters
    ----------
    indices_by_fold : dict
        dict returned from get_fold_indices.
    current_fold : int
        zero-based fold number in range of cross-validation folds.

    Returns
    -------
    i_train : list
        DESCRIPTION.
    i_val : list
        DESCRIPTION.

    """
    i_train = []
    i_val = []
    
    for k in indices_by_fold.keys():
        if k == current_fold:
            i_val.extend(indices_by_fold[k])
        else:
            i_train.extend(indices_by_fold[k])
    
    return i_train, i_val


############
#  Models  #
############

def get_model():
    return multi_class_unet_model(n_classes=n_classes, IMG_HEIGHT=SIZE_Y, IMG_WIDTH=SIZE_X, IMG_CHANNELS=IMG_CHANNELS)

IMG_CHANNELS = X_train.shape[3]
model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

##########################################
#  Tune batch size and number of epochs  #
##########################################

study_folder = 'batch_size_epoch_study/'

#create directories
study_dir = os.path.dirname(study_folder)
if not os.path.isdir(study_dir):
    os.makedirs(study_dir)
    os.makedirs(study_dir+'/logs/')
    os.makedirs(study_dir+'/models/')
    

batch_size = [64]# [8, 16, 32, 64]
epochs = [100] #[10, 50, 100]

#get cross-validation indices dictionary
indices_by_fold = get_fold_indices(X_train.shape[0], folds)

for b in batch_size:
    for e in epochs:
        for f in [0]:#range(folds):
            print(f"batch size: {b}, epochs: {e}, fold: {f}")
            #get cross-validation indices
            i_train, i_val = get_train_val_fold_split(indices_by_fold, f)
            
            #callbacks
            csv_logger = CSVLogger(study_folder + f'logs/B{b}_E{e}_F{f}.log', separator=',', append=True)
    
            #train
            history = model.fit(X_train[i_train], y_train_cat[i_train],
                                batch_size=b,
                                verbose=2,
                                epochs=e,
                                validation_data=(X_train[i_val], y_train_cat[i_val]),
                                class_weight=class_weights_dict,
                                shuffle=False,
                                callbacks=[csv_logger])

            #save model
            model.save(study_folder + f"models/B{b}_E{e}_F{f}.keras")
            print("model saved...")


