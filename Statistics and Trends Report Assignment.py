# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 18:33:59 2022

@author: ag11afr
"""

#import required modules

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


# Make datafrane (df_all_data) for climate change large data set from xlsx file named appropriatly

df_all_data = pd.read_excel("Climate_Change_Data_Set.xlsx")  



print(df_all_data)
