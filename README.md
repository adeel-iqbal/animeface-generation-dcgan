# ×͜× Anime Face Generation using DCGAN

A Deep Convolutional Generative Adversarial Network (DCGAN) implementation that generates unique anime-style character faces from random noise vectors. Built with TensorFlow and Keras, this project demonstrates the power of adversarial training to create high-quality synthetic images.

## Overview

This project trains a DCGAN model on the Anime Faces dataset to generate 64×64 pixel anime character portraits. The model learns to create diverse and realistic anime faces by pitting two neural networks against each other: a Generator that creates images and a Discriminator that evaluates their authenticity.

## Features

- **DCGAN Architecture**: Implements the proven Deep Convolutional GAN design with transposed convolutions for upsampling
- **Progressive Upsampling**: Generator builds images from 4×4 to 64×64 resolution through multiple layers
- **Stable Training**: Uses LeakyReLU activations, batch normalization, and Adam optimizer with tuned hyperparameters
- **Visual Progress Tracking**: Generates sample images every 5 epochs to monitor training evolution
- **Easy Generation**: Simple function to create new anime faces on demand after training

## Architecture

### Generator
The Generator takes a 100-dimensional noise vector and progressively upsamples it through four Conv2DTranspose layers:
- Input: 100-dimensional random noise
- Layer 1: Dense → 4×4×1024
- Layer 2: Conv2DTranspose → 8×8×512
- Layer 3: Conv2DTranspose → 16×16×256
- Layer 4: Conv2DTranspose → 32×32×128
- Output: Conv2DTranspose → 64×64×3 (RGB image)

### Discriminator
The Discriminator evaluates images through four convolutional layers with downsampling:
- Input: 64×64×3 RGB image
- Convolutional layers progressively reduce spatial dimensions
- Dropout layer prevents overfitting
- Output: Single sigmoid value (0 = fake, 1 = real)

## Dataset

This project uses the [Anime Faces dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) from Kaggle, containing over 43,000 anime character face images.

## Requirements

```
tensorflow>=2.x
matplotlib
numpy
kaggle
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/adeel-iqbal/animeface-generation-dcgan.git
cd animeface-generation-dcgan
```

2. Set up Kaggle API credentials:
   - Download your `kaggle.json` from Kaggle account settings
   - Place it in the project directory

3. Install dependencies:
```bash
pip install tensorflow matplotlib numpy kaggle
```

## Usage

### Training the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook anime_generation_dcgan.ipynb
```

The notebook will:
1. Download and prepare the Anime Faces dataset
2. Build the Generator and Discriminator networks
3. Train for 30 epochs (customizable)
4. Save progress images every 5 epochs
5. Save the final trained Generator model

### Generating New Faces

After training, generate new anime faces using:

```python
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained generator
generator = tf.keras.models.load_model('anime_generator_final.keras')

# Generate a new face
seed = tf.random.normal([1, 100])
prediction = generator(seed, training=False)

# Display the result
plt.imshow((prediction[0].numpy() * 127.5 + 127.5).astype("uint8"))
plt.axis('off')
plt.show()
```

## Training Configuration

- **Image Size**: 64×64 pixels
- **Batch Size**: 64
- **Epochs**: 30
- **Optimizer**: Adam (learning rate: 0.0002, beta_1: 0.5)
- **Loss Function**: Binary Cross-Entropy

## Results

The model progressively learns to generate anime faces over 30 epochs. Early epochs show noise and basic shapes, while later epochs produce detailed features including eyes, hair, and facial expressions characteristic of anime art style.

## Project Structure

```
animeface-generation-dcgan/
├── anime_generation_dcgan.ipynb  # Main training notebook
├── README.md                      # Project documentation
```

## How It Works

**Generative Adversarial Networks (GANs)** work through an adversarial process:

1. **Generator** creates fake images from random noise
2. **Discriminator** tries to distinguish real images from fake ones
3. Both networks improve through competition:
   - Generator learns to create more realistic images
   - Discriminator learns to better identify fakes
4. Training continues until the Generator produces convincing results

## Future Improvements

- Implement Progressive Growing GAN for higher resolution outputs
- Add conditional generation to control specific features (hair color, eye style, etc.)
- Experiment with StyleGAN architecture for improved quality
- Create a web interface for easy image generation

## License

This project is open source and available for educational purposes.

## Contact

**Adeel Iqbal Memon**

- Email: adeelmemon096@yahoo.com
- LinkedIn: [linkedin.com/in/adeeliqbalmemon](https://linkedin.com/in/adeeliqbalmemon)
- GitHub: [@adeel-iqbal](https://github.com/adeel-iqbal)

## Acknowledgments

- Dataset: [Anime Faces on Kaggle](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) by Soumik Rakshit
- Inspired by the original DCGAN paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

---

⭐ If you find this project helpful, please consider giving it a star!
