#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
get_ipython().system('pip install scikit-learn')
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize


# In[2]:


get_ipython().system('pip install Pillow')
# Define the paths to your dataset directories
clean_images_dir = 'data/train/clean/'
noisy_images_dir = 'data/train/noisy/'

# Define desired image size (adjust as needed)
image_size = (128, 128)


def load_images_from_folder(folder, size):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path is not None:
            img = load_img(img_path, target_size=size, color_mode='rgb')
            img = img_to_array(img)
            images.append(img)
    return np.array(images)

get_ipython().system('pip install Pillow')


# Load and preprocess the data
clean_images = load_images_from_folder(clean_images_dir, image_size)
noisy_images = load_images_from_folder(noisy_images_dir, image_size)


# In[3]:


# Normalize the images to [0, 1]
clean_images_normalized = normalize(clean_images, axis=1)
noisy_images_normalized = normalize(noisy_images, axis=1)


# In[4]:


# Split the dataset into training and testing sets (adjust split ratio as needed)
x_train_noisy, x_test_noisy, x_train_clean, x_test_clean = train_test_split(
    noisy_images_normalized, clean_images_normalized, test_size=0.2, random_state=42
)


# In[5]:





# In[6]:


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model


# In[ ]:




