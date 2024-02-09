
#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from PIL import Image

# Define the paths to your dataset directories
clean_images_dir = 'data/train/clean/'
noisy_images_dir = 'data/train/noisy/'

# Define desired image size (adjust as needed)
image_size = (128, 128)

def load_images_from_folder(folder, size):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=size, color_mode='rgb')
        img = img_to_array(img)
        images.append(img)
    return np.array(images)

# Load and preprocess the data
clean_images = load_images_from_folder(clean_images_dir, image_size)
noisy_images = load_images_from_folder(noisy_images_dir, image_size)
clean_images_normalized = normalize(clean_images, axis=1)
noisy_images_normalized = normalize(noisy_images, axis=1)

# Split the dataset
x_train_noisy, x_test_noisy, x_train_clean, x_test_clean = train_test_split(
    noisy_images_normalized, clean_images_normalized, test_size=0.2, random_state=42
)

print(f"Training data (noisy): {x_train_noisy.shape}")
print(f"Training data (clean): {x_train_clean.shape}")
print(f"Test data (noisy): {x_test_noisy.shape}")
print(f"Test data (clean): {x_test_clean.shape}")

# Autoencoder Model
def build_autoencoder(img_shape):
    input_img = Input(shape=img_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

img_shape = x_train_noisy.shape[1:]
autoencoder = build_autoencoder(img_shape)
autoencoder.summary()

# Train the Model
history = autoencoder.fit(x_train_noisy, x_train_clean, epochs=50, batch_size=128, shuffle=True,
                          validation_data=(x_test_noisy, x_test_clean), verbose=1)

# Save the trained model
autoencoder.save('autoencoder_model.h5')

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

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

# Load Test Images, Predict, and Save/Display Denoised Images
test_noisy_images, filenames = load_images_from_folder('data/test/', size=(128, 128))
test_noisy_images = test_noisy_images.astype('float32') / 255.0
denoised_images = autoencoder.predict(test_noisy_images)

# Display Denoised Images
def display_images(noisy_images, denoised_images, n=5):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(noisy_images[i])
        plt.title("Noisy")


from PIL import Image
def save_images(images, filenames, save_dir='data/denoised/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img, filename in zip(images, filenames):
        img = Image.fromarray(np.uint8(img * 255))
        img.save(os.path.join(save_dir, filename))

# Save the denoised images
save_images(denoised_images, filenames)
