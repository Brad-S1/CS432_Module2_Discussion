#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 22:16:33 2025

@author: bradsommer
"""

import pandas as pd
from sklearn.cluster import KMeans
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

# Set the qualitative variables as data type 'category.
correctDtypes_df = original_df.copy()
correctDtypes_df['region'] = correctDtypes_df['region'].astype('category')
correctDtypes_df['label'] = correctDtypes_df['label'].astype('category')
print(correctDtypes_df.dtypes)

# drop qualitative columns
quant_df = correctDtypes_df.copy()
quant_df=quant_df.drop(['label', 'region'], axis=1)
print(quant_df)

## Instatiate kmeans
kmeans_objectCustomers = KMeans(n_clusters=3, max_iter=100)
kmeans_clusters = kmeans_objectCustomers.fit(quant_df)

# get cluster labels
labels_clusters = kmeans_clusters.labels_
print(labels_clusters)

# add cluster assignments back into data for later use
cluster_df = quant_df.copy()
cluster_df['kmeans_cluster'] = labels_clusters
print(cluster_df)

# create a pairplot of all variables colored by cluster
sns.pairplot(cluster_df, hue='kmeans_cluster', palette='viridis', 
             vars=['age', 'income', 'monthly_purchases'])
plt.suptitle('Cluster Visualization', y=1.02, fontsize=16)
plt.show()

