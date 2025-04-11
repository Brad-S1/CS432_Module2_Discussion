#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 22:16:33 2025

@author: bradsommer
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filepath = '/Users/bradsommer/Documents/School/OSU/CS 432/Module 2/Discussion/customer_dataset.csv'

original_df = pd.read_csv(filepath)
print(original_df)

# Display basic information about the dataframe structure
original_df.info()
# Display data types:
print("Dataset data types:\n")
print(original_df.dtypes)
# Display all column names
print("Number of columns:", len(original_df.columns))
print("\nColumn names:")
print(original_df.columns.tolist())

correctDtypes_df = original_df.copy()
# Set the qualitative variables as data type 'category.
correctDtypes_df['region'] = correctDtypes_df['region'].astype('category')
correctDtypes_df['label'] = correctDtypes_df['label'].astype('category')
print(correctDtypes_df.dtypes)