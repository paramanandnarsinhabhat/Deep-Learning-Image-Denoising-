
# Image Denoising with Deep Learning

This repository contains the implementation of an autoencoder for the purpose of image denoising. The project uses a Convolutional Neural Network (CNN) based autoencoder trained on noisy images to predict cleaner versions of the images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The following libraries are required to run the code:

- Numpy 1.22.3
- TensorFlow 2.8.0
- Scikit-learn 1.0.2
- Matplotlib 3.5.1
- Pillow 9.0.1

You can install these packages by running:

```
pip install -r requirements.txt
```

### Installing

First, clone this repository to your local machine.

```
git clone https://github.com/paramanandnarsinhabhat/Deep-Learning-Image-Denoising-.git
```

Next, download the dataset from the link below and extract it into the `data` directory in the cloned repository.

[Download Dataset](https://import.cdn.thinkific.com/118220/image_denoising_dataset-200512-163438.zip)

Ensure the following directory structure:

```
data/
│
├── denoised/
├── test/
├── train/
│   ├── clean/
│   └── noisy/
└── ...
```

### Training the Autoencoder

To train the autoencoder, navigate to the source directory and run:

```
python denoised.py
```

The script will train the autoencoder model using the noisy images as input and the corresponding clean images as output.

### Testing the Model

After training, you can test the autoencoder's performance on the test set of noisy images to see how well it can remove noise.

## Results

The trained model will output denoised images which you can visually inspect to assess the denoising quality.

## Authors

* **Paramanand Bhat

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

