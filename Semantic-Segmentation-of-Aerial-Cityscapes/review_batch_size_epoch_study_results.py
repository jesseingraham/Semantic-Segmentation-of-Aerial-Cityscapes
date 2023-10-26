# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:52:06 2023

@author: Kaitlin
"""

import pandas as pd
from matplotlib import pyplot as plt
import glob
import os



study_folder = 'batch_size_epoch_study/'

logs = glob.glob(os.path.join(study_folder, 'logs/*.log'))

b8df = pd.DataFrame()
for log in logs:
    temp_df = pd.read_csv(os.path.join(study_folder, 'logs/', os.path.basename(log)))
    if logs.index(log) == 0:
        b8df = temp_df.copy()
    else:
        b8df.join(temp_df, on='epoch', rsuffix=os.path.basename(log).split('.')[0])