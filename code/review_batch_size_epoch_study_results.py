# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:52:06 2023

@author: Kaitlin
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import glob
import os

study_folder = 'batch_size_epoch_study/'


###########################
#  Combine log file data  #
###########################

logs = glob.glob(os.path.join(study_folder, 'logs/*.log'))

study_df = pd.DataFrame()
for log in logs:
    base = os.path.basename(log)
    batch_size = base.split('_')[0][1:]
    n_epochs = base.split('_')[1][1:]
    fold = (base.split('_')[2]).split('.')[0][1:]
    
    temp_df = pd.read_csv(os.path.join(study_folder, 'logs/', os.path.basename(log)))
    temp_df['batch_size'] = batch_size
    temp_df['n_epochs'] = n_epochs
    temp_df['fold'] = fold

    if logs.index(log) == 0:
        study_df = temp_df.copy()  # initialize by copying
    else:
        study_df = pd.concat([study_df, temp_df], axis=0)


#################################
#  Organize dataframes to plot  #
#################################

study_df.rename({'accuracy' : 'train_accuracy', 'loss' : 'train_loss'}, axis=1, inplace=True)

loss_df = study_df.drop(['train_accuracy', 'val_accuracy'], axis=1)
loss_df.rename({'train_loss' : 'train', 'val_loss' : 'val'}, axis=1, inplace=True)
loss_df = loss_df.melt(id_vars=['epoch', 'batch_size', 'n_epochs', 'fold'], value_vars=['train', 'val'], var_name='loss')

accuracy_df = study_df.drop(['train_loss', 'val_loss'], axis=1)
accuracy_df.rename({'train_accuracy' : 'train', 'val_accuracy' : 'val'}, axis=1, inplace=True)
accuracy_df = accuracy_df.melt(id_vars=['epoch', 'batch_size', 'n_epochs', 'fold'], value_vars=['train', 'val'], var_name='accuracy')


##########
#  Plot  #
##########

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,8))
sns.lineplot(ax=axs[0], data=loss_df, x='epoch', y='value', style='batch_size', hue='loss')
sns.lineplot(ax=axs[1], data=accuracy_df, x='epoch', y='value', style='batch_size', hue='accuracy')

axs[0].set_ylabel('loss')
axs[0].set_title('Training and Validation Loss')
axs[1].set_ylabel('accuracy')
axs[1].set_title('Training and Validation Accuracy')

for ax in axs:
    ax.grid(True)

    #place legend outside center right border of plot
    ax.legend(bbox_to_anchor=(1.02, 0.7), loc='upper left', borderaxespad=0)

plt.show()
