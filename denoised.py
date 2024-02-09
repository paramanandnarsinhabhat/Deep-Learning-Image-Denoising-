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


# In[8]:


print(f"Training data (noisy): {x_train_noisy.shape}")  # e.g., (num_samples, 128, 128, 3)
print(f"Training data (clean): {x_train_clean.shape}")  # e.g., (num_samples, 128, 128, 3)
print(f"Test data (noisy): {x_test_noisy.shape}")       # e.g., (num_samples, 128, 128, 3)
print(f"Test data (clean): {x_test_clean.shape}")       # e.g., (num_samples, 128, 128, 3)


# In[9]:


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# Assuming x_train_noisy and x_train_clean have the same shape
img_shape = x_train_noisy.shape[1:]  # This takes the shape of the data excluding the batch dimension
def build_autoencoder(img_shape):
    input_img = Input(shape=img_shape)
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder


# In[10]:


autoencoder = build_autoencoder(img_shape)
autoencoder.summary()


# In[11]:


# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')


# In[12]:


# Train the model
history = autoencoder.fit(
    x_train_noisy, x_train_clean,  # Using noisy images as input and clean images as target
    epochs=50,  # Number of epochs to train for
    batch_size=128,  # Batch size for training
    shuffle=True,  # Shuffle the training data
    validation_data=(x_test_noisy, x_test_clean),  # Validation data for monitoring
    verbose=1  # Show training log
)


# In[14]:


# Optional: Save the trained model
autoencoder.save('autoencoder_model.h5')


# In[16]:


get_ipython().system('pip install matplotlib')
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


# In[17]:


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss History (Log Scale)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()

plt.tight_layout()
plt.show()


# In[18]:


#Step 1: Load the Noisy Test Images
def load_images_from_folder(folder, size=(128, 128)):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path is not None:
            img = load_img(img_path, target_size=size, color_mode='rgb')
            img = img_to_array(img)
            images.append(img)
            filenames.append(filename)
    return np.array(images), filenames


# In[20]:


# Adjust the path and size as necessary
test_noisy_images, filenames = load_images_from_folder('data/test/', size=(128, 128))
test_noisy_images = test_noisy_images.astype('float32') / 255.0


# In[21]:


#Step 2: Use the Model to Denoise the Test Images
# Assuming your autoencoder model is loaded and named 'autoencoder'
denoised_images = autoencoder.predict(test_noisy_images)


# In[22]:


#3. Visualize or Save the Denoised Images
def display_images(noisy_images, denoised_images, n=5):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original noisy image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(noisy_images[i])
        plt.title("Noisy")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display denoised image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(denoised_images[i])
        plt.title("Denoised")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# Display the first 5 noisy and denoised images for comparison
display_images(test_noisy_images, denoised_images, n=5)


# In[24]:


from PIL import Image
def save_images(images, filenames, save_dir='data/denoised/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img, filename in zip(images, filenames):
        img = Image.fromarray(np.uint8(img * 255))
        img.save(os.path.join(save_dir, filename))

# Save the denoised images
save_images(denoised_images, filenames)


# In[ ]:




