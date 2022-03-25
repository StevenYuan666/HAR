#!/usr/bin/env python
# coding: utf-8

# # import needed packages

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
import os
import csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import scipy.stats as stats
from sklearn.model_selection import train_test_split


# # Data pre-processing

# In[2]:


path_phone_accel = "/Users/stevenyuan/Documents/McGill/Research/HAR/wisdm-dataset/raw/phone/accel/"
files_phone_accel = os.listdir(path_phone_accel)
path_phone_gyro = "/Users/stevenyuan/Documents/McGill/Research/HAR/wisdm-dataset/raw/phone/gyro/"
files_phone_gyro = os.listdir(path_phone_gyro)
path_watch_accel = "/Users/stevenyuan/Documents/McGill/Research/HAR/wisdm-dataset/raw/watch/accel/"
files_watch_accel = os.listdir(path_watch_accel)
path_watch_gyro = "/Users/stevenyuan/Documents/McGill/Research/HAR/wisdm-dataset/raw/watch/gyro/"
files_watch_gyro = os.listdir(path_watch_gyro)


# In[89]:


files_phone_accel.sort()
files_phone_gyro.sort()
files_watch_accel.sort()
files_watch_gyro.sort()
files_phone_accel = files_phone_accel[1:]
files_phone_gyro = files_phone_gyro[1:]
files_watch_accel = files_watch_accel[1:]
files_watch_gyro = files_watch_gyro[1:]


# In[90]:


# phone accelerator
df_phone_accel = pd.DataFrame()
for file in files_phone_accel:
    temp_df = pd.read_csv(path_phone_accel + file, sep=",", header=None)
    temp_df.columns = ["id", "actCode", "Timestamp", "a_x", "a_y", "a_z"]
    temp_df['a_z'] = temp_df['a_z'].str.replace(';', '')
    df_phone_accel = pd.concat([df_phone_accel, temp_df], sort=False)
print(df_phone_accel)


# In[91]:


# phone gyroscope
df_phone_gyro = pd.DataFrame()
for file in files_phone_gyro:
    temp_df = pd.read_csv(path_phone_gyro + file, sep=",")
    temp_df.columns = ["id", "actCode", "Timestamp", "g_x", "g_y", "g_z"]
    temp_df['g_z'] = temp_df['g_z'].str.replace(';', '')
    df_phone_gyro = pd.concat([df_phone_gyro, temp_df], sort=False)
print(df_phone_gyro)


# In[92]:


# smartwatch accelerator
df_watch_accel = pd.DataFrame()
for file in files_watch_accel:
    temp_df = pd.read_csv(path_watch_accel + file, sep=",")
    temp_df.columns = ["id", "actCode", "Timestamp", "a_x", "a_y", "a_z"]
    temp_df['a_z'] = temp_df['a_z'].str.replace(';', '')
    df_watch_accel = pd.concat([df_watch_accel, temp_df], sort=False)
print(df_watch_accel)


# In[93]:


# smartwatch gyroscope
df_watch_gyro = pd.DataFrame()
for file in files_watch_gyro:
    temp_df = pd.read_csv(path_watch_gyro + file, sep=",")
    temp_df.columns = ["id", "actCode", "Timestamp", "g_x", "g_y", "g_z"]
    temp_df['g_z'] = temp_df['g_z'].str.replace(';', '')
    df_watch_gyro = pd.concat([df_watch_gyro, temp_df], sort=False)
print(df_watch_gyro)


# In[94]:


# do the one-hot encoding for the labels
onehot_encoder = OneHotEncoder()
label_encoder = LabelEncoder()
df_phone_accel['label'] = label_encoder.fit_transform(df_phone_accel['actCode'])
df_phone_gyro['label'] = label_encoder.fit_transform(df_phone_gyro['actCode'])
df_watch_accel['label'] = label_encoder.fit_transform(df_watch_accel['actCode'])
df_watch_gyro['label'] = label_encoder.fit_transform(df_watch_gyro['actCode'])


# In[95]:


# standarize the value of each axis
scaler = StandardScaler()
df_phone_accel[['a_x', 'a_y', 'a_z']] = scaler.fit_transform(df_phone_accel[['a_x', 'a_y', 'a_z']])
df_phone_gyro[['g_x', 'g_y', 'g_z']] = scaler.fit_transform(df_phone_gyro[['g_x', 'g_y', 'g_z']])
df_watch_accel[['a_x', 'a_y', 'a_z']] = scaler.fit_transform(df_watch_accel[['a_x', 'a_y', 'a_z']])
df_watch_gyro[['g_x', 'g_y', 'g_z']] = scaler.fit_transform(df_watch_gyro[['g_x', 'g_y', 'g_z']])


# In[96]:


print(df_phone_accel)


# In[97]:


print(df_phone_gyro)


# In[98]:


print(df_watch_accel)


# In[99]:


print(df_watch_gyro)


# In[100]:


df_phone_accel_x, df_phone_accel_y = df_phone_accel[['a_x', 'a_y', 'a_z']], df_phone_accel['label']


# In[101]:


df_phone_gyro_x, df_phone_gyro_y = df_phone_gyro[['g_x', 'g_y', 'g_z']], df_phone_gyro['label']


# In[102]:


df_watch_accel_x, df_watch_accel_y = df_watch_accel[['a_x', 'a_y', 'a_z']], df_watch_accel['label']


# In[103]:


df_watch_gyro_x, df_watch_gyro_y = df_watch_gyro[['g_x', 'g_y', 'g_z']], df_watch_gyro['label']


# In[104]:


df_phone_accel_x_train, df_phone_accel_x_test, df_phone_accel_y_train, df_phone_accel_y_test = train_test_split(df_phone_accel_x, df_phone_accel_y, test_size = 0.2)


# In[105]:


print(df_phone_accel_x_train)
print(df_phone_accel_x_test)
print(df_phone_accel_y_train)
print(df_phone_accel_y_test)


# In[106]:


df_phone_gyro_x_train, df_phone_gyro_x_test, df_phone_gyro_y_train, df_phone_gyro_y_test = train_test_split(df_phone_gyro_x, df_phone_gyro_y, test_size = 0.2)


# In[107]:


print(df_phone_gyro_x_train)
print(df_phone_gyro_x_test)
print(df_phone_gyro_y_train)
print(df_phone_gyro_y_test)


# In[108]:


df_watch_accel_x_train, df_watch_accel_x_test, df_watch_accel_y_train, df_watch_accel_y_test = train_test_split(df_watch_accel_x, df_watch_accel_y, test_size = 0.2)


# In[109]:


print(df_watch_accel_x_train)
print(df_watch_accel_x_test)
print(df_watch_accel_y_train)
print(df_watch_accel_y_test)


# In[110]:


df_watch_gyro_x_train, df_watch_gyro_x_test, df_watch_gyro_y_train, df_watch_gyro_y_test = train_test_split(df_watch_gyro_x, df_watch_gyro_y, test_size = 0.2)


# In[111]:


print(df_watch_gyro_x_train)
print(df_watch_gyro_x_test)
print(df_watch_gyro_y_train)
print(df_watch_gyro_y_test)


# In[ ]:




