# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 07:01:46 2023

@author: Kaitlin

This script will load image data 

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



model_to_load = 'UNET_Multiclass'  #"UNET_Multiclass", "DeepLabV3Plus"

#data loading
from functions.load_train_test_data import load_train_test_data

from keras.models import load_model
import os
import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#saved models directory
saved_models_path = model_to_load + "_trained_models/"
saved_models_dir = os.path.dirname(saved_models_path)

#load weights from previous training here, if any
models_list = glob.glob(os.path.join(saved_models_dir, "*" + ".keras"))

#get model epochs
models_list = [os.path.basename(path) for path in models_list]  #reduce to base name

#load model
#model = load_model(os.path.join(saved_models_dir, models_list[0]))
model = load_model("batch_size_epoch_study/models/B8_E100_F0.keras")

#load training history
#history = pd.read_csv(os.path.join('logs/', model_to_load+'_training.log'))
history = pd.read_csv("batch_size_epoch_study/logs/B8_E100_F0.log")

####################
#  Training Plots  #
####################

#plot the training and validation loss at each epoch
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plot the training and validation accuracy at each epoch
acc = history['accuracy']
val_acc = history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


###################
#  Get test data  #
###################

X_train, y_train_cat, X_test, y_test_cat, n_classes, class_weights_dict = load_train_test_data(image_path=image_path, image_type=image_type, mask_path=mask_path, mask_type=mask_type, sizey=SIZE_Y, sizex=SIZE_X)


#############################
#  Intersection over Union  #
#############################

#Using built in keras function
from keras.metrics import MeanIoU
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
n_classes = 8
IOU_keras = MeanIoU(num_classes=n_classes)
y_test = np.argmax(y_test_cat, axis=3)
y_test = np.expand_dims(y_test, axis=3)
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[0,5] + values[0,6] + values[0,7] + \
                          values[1,0] + values[2,0] + values[3,0] + values[4,0] + values[5,0] + values[6,0] + values[7,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[1,5] + values[1,6] + values[1,7] + \
                          values[0,1] + values[2,1] + values[3,1] + values[4,1] + values[5,1] + values[6,1] + values[7,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4] + values[2,5] + values[2,6] + values[2,7] + \
                          values[0,2] + values[1,2] + values[3,2] + values[4,2] + values[5,2] + values[6,2] + values[7,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[3,5] + values[3,6] + values[3,7] + \
                          values[0,3] + values[1,3] + values[2,3] + values[4,3] + values[5,3] + values[6,3] + values[7,3])
class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3] + values[4,5] + values[4,6] + values[4,7] + \
                          values[0,4] + values[1,4] + values[2,4] + values[3,4] + values[5,4] + values[6,4] + values[7,4])
class6_IoU = values[5,5]/(values[5,5] + values[5,0] + values[5,1] + values[5,2] + values[5,3] + values[5,4] + values[5,6] + values[5,7] + \
                          values[0,5] + values[1,5] + values[2,5] + values[3,5] + values[4,5] + values[6,5] + values[7,5])
class7_IoU = values[6,6]/(values[6,6] + values[6,0] + values[6,1] + values[6,2] + values[6,3] + values[6,4] + values[6,5] + values[6,7] + \
                          values[0,6] + values[1,6] + values[2,6] + values[3,6] + values[4,6] + values[5,6] + values[7,6])
class8_IoU = values[7,7]/(values[7,7] + values[7,0] + values[7,1] + values[7,2] + values[7,3] + values[7,4] + values[7,5] + values[7,6] + \
                          values[0,7] + values[1,7] + values[2,7] + values[3,7] + values[4,7] + values[5,7] + values[6,7])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)
print("IoU for class5 is: ", class5_IoU)
print("IoU for class6 is: ", class6_IoU)
print("IoU for class7 is: ", class7_IoU)
print("IoU for class8 is: ", class8_IoU)


#############################
#  Predict on a few images  #
#############################

import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()






















