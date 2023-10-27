# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 08:31:06 2023

@author: Kaitlin

NOTE: Running this file will clear all variables in memory!

NOTE: 50 epochs took 4.76 hours to run

"""

##############################
#  Data loading user inputs  #
##############################

#new image size
SIZE_X = 512
SIZE_Y = 512

#filepaths
image_path = "../data/Master/src/"
image_type = ".jpeg"

mask_path = "../data/Master/gt/"
mask_type = ".bmp"


################################
#  Model training user inputs  #
################################

#declare model
chosen_model = 'UNET_Multiclass'  #"UNET_Multiclass", "DeepLabV3Plus"

#training
N_EPOCHS = 50  # total number of epochs to be trained (even if continuing training)
load_model = True  #bool


##############
#  Imports   #
##############

#data loading
from functions.load_train_test_data import load_train_test_data

#models and training
from segmentation_models import multi_class_unet_model #Uses softmax
from keras.models import load_model
from keras.callbacks import CSVLogger

#other
import os
import glob


##############
#  Get data  #
##############

X_train, y_train_cat, X_test, y_test_cat, n_classes, class_weights_dict = load_train_test_data(image_path=image_path, image_type=image_type, mask_path=mask_path, mask_type=mask_type, sizey=SIZE_Y, sizex=SIZE_X)


############
#  Models  #
############
def get_model():
    return multi_class_unet_model(n_classes=n_classes, IMG_HEIGHT=SIZE_Y, IMG_WIDTH=SIZE_X, IMG_CHANNELS=IMG_CHANNELS)


####################################
#  Load, compile, and train model  #
####################################

#check training data
assert X_train.shape[1] == SIZE_Y
assert X_train.shape[2] == SIZE_X
IMG_CHANNELS = X_train.shape[3]
print(f"The model is training on {X_train.shape[0]} images of size {SIZE_Y} by {SIZE_X} with {IMG_CHANNELS} channel(s).")

#saved models directory
saved_models_path = chosen_model + "_trained_models/"
saved_models_dir = os.path.dirname(saved_models_path)
if not os.path.isdir(saved_models_dir):
    os.makedirs(saved_models_dir)

#load weights from previous training here, if any
largest_epoch = 0  #overwritten if model loaded
if load_model:
    # get latest model
    models_list = glob.glob(os.path.join(saved_models_dir, "*" + ".keras"))
    if models_list:
        # get model with greatest epoch
        models_list = [os.path.basename(path) for path in models_list]
        s_before = [m[:m.find('_epochs')] for m in models_list]
        saved_model_epochs = [int(m[m.rfind('_')+1:]) for m in s_before]
        largest_epoch = max(saved_model_epochs)
        latest_model_index = saved_model_epochs.index(largest_epoch)
        latest_model = models_list[latest_model_index]
        model = load_model(os.path.join(saved_models_dir, latest_model))
        print('model restored... training continued from epoch {}...'.format(largest_epoch))
    else:
        print('model not found... creating new model...')
        model = get_model()
        print('new model created...')

        #compile and print model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
else:
    #get new model
    model = get_model()
    print('new model created...')

    #compile and print model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    

#callbacks
csv_logger = CSVLogger(os.path.join('logs/',chosen_model+'_training.log'), separator=',', append=True)

#train
history = model.fit(X_train, y_train_cat,
                    batch_size = 16,
                    verbose=1,
                    initial_epoch=largest_epoch,
                    epochs=N_EPOCHS,
                    validation_data=(X_test, y_test_cat),
                    class_weight=class_weights_dict,
                    shuffle=False,
                    callbacks=[csv_logger])

################
#  save model  #
################

model_name = chosen_model + "_{}_epochs".format(N_EPOCHS + largest_epoch)
model.save(os.path.join(saved_models_dir, model_name + ".keras"))
print("model saved...")





















