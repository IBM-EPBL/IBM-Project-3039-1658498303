#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the ImageDataGenerator library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


import os
for dirname, _, filenames in os.walk(r'C:\Users\HP\Desktop\dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
print(tf.__version__)


# In[5]:


train_dir = Path(r'C:\Users\HP\Desktop\dataset\TRAIN_SET\TRAIN_SET')
train_filepaths = list(train_dir.glob(r'**/*.jpg'))

test_dir = Path(r'C:\Users\HP\Desktop\dataset\TEST_SET-20221101T044129Z-001\TEST_SET')
test_filepaths = list(test_dir.glob(r'**/*.jpg')) 


# In[6]:


def image_processing(filepath):
    """ Create a DataFrame with the filepath and the labels of the pictures
    """
    
    labels = [str(filepath[i]).split("\\")[-2]               for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop = True)
    
    return df


# In[7]:


train_df = image_processing(train_filepaths)
test_df = image_processing(test_filepaths)
train_df.head(5)


# In[8]:


df_unique = train_df.copy().drop_duplicates(subset=["Label"]).reset_index()
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(8, 7),
                        subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df_unique.Filepath[i]))
    ax.set_title(df_unique.Label[i], fontsize = 12)
plt.tight_layout(pad=0.4)
plt.show()


# In[9]:


#Import the ImageDataGenerator library
#Configure ImageDataGenerator class
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)


# In[10]:


#ApplyImageDataGenerator functionality to Trainset and Testset
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=0,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)


# In[11]:


#ApplyImageDataGenerator functionality to Trainset and Testset
test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)


# In[ ]:




