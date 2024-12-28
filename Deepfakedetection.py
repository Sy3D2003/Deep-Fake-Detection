import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from skimage.io import imread
from skimage.transform import resize

# At the beginning of your script
np.random.seed(42)
tf.random.set_seed(42)



results_summary = {
    'total_evaluated': 0,
    'total_fake': 0,
    'total_real': 0,
    'uncertain': 0
}


# Constants for the GAN
BATCH_SIZE = 1
NOISE_DIM = 100
EPOCHS = 100
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 64, 64, 3  # Adjust to your image size
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Function to load and preprocess an image
def load_image(image_path, image_size):
    image = imread(image_path)
    # If the image has an alpha channel, take only the first three channels
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = resize(image, image_size, anti_aliasing=True)
    image = (image - 0.5) * 2.0  # Normalize to [-1, 1]
    return image


# Load and preprocess the images
real_image_path = 'D:/OneDrive - student.newinti.edu.my/Desktop/web/real1.png'
fake_image_path = 'D:/OneDrive - student.newinti.edu.my/Desktop/web/deepfake1.png'
real_image = load_image(real_image_path, (IMG_HEIGHT, IMG_WIDTH))
fake_image = load_image(fake_image_path, (IMG_HEIGHT, IMG_WIDTH))

# Expand dimensions to have a batch size of 1
real_image = np.expand_dims(real_image, axis=0)
fake_image = np.expand_dims(fake_image, axis=0)

def build_generator(z_dim):
    model = tf.keras.Sequential()

    # Start with a fully connected layer
    model.add(layers.Dense(512 * 4 * 4, input_dim=z_dim))
    model.add(layers.Reshape((4, 4, 512)))

    # Up-sample to 8x8
    model.add(layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Up-sample to 16x16
    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Up-sample to 32x32
    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Up-sample to 64x64
    model.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same'))
    model.add(layers.Activation('tanh'))

    return model


# Discriminator Model
def build_discriminator():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=IMG_SHAPE))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])

# Build the generator
z_dim = 100  # This should be the length of the input noise vector
generator = build_generator(z_dim)



# The combined model (stacked generator and discriminator)
discriminator.trainable = False
combined = models.Sequential()
combined.add(generator)
combined.add(discriminator)
combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# Training loop
for epoch in range(EPOCHS):
    
    # Train Discriminator
    d_loss_real = discriminator.train_on_batch(real_image, np.ones((BATCH_SIZE, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_image, np.zeros((BATCH_SIZE, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    g_loss = combined.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))

    # Progress logging
    print(f"Epoch {epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

# After training, you can generate images and save models as needed.
# Function to evaluate an image with the discriminator

 

def evaluate_image(image_path, discriminator, image_size):
    global results_summary
    try:
        image = load_image(image_path, image_size)
        image_batch = np.expand_dims(image, axis=0)
        prediction = discriminator.predict(image_batch)
        results_summary['total_evaluated'] += 1
        
        # Display the image with prediction
        plt.imshow(image * 0.5 + 0.5)  # Rescale the image values from [-1, 1] to [0, 1]
        plt.title(f"Prediction: {prediction[0][0]:.4f}")
        plt.axis('off')
        plt.show(block=False)  # Set block to False to allow continuation
        
        # Set a pause duration to display each image for a short time
        plt.pause(1)  # Adjust the time as needed
        plt.close()  # Close the image window

        # Determine if the image is fake or real
        if prediction < 0.3:
            print(f"The image at {image_path} is likely fake.")
            results_summary['total_fake'] += 1
        elif prediction > 0.7:
            print(f"The image at {image_path} is likely real.")
            results_summary['total_real'] += 1
        else:
            print(f"The discriminator is uncertain about the image at {image_path}.")
            results_summary['uncertain'] += 1
    except Exception as e:
        print(f"Could not process image at {image_path}: {e}")


# Path to the directory with test images
test_images_directory = 'D:/OneDrive - student.newinti.edu.my/Desktop/web'

# Evaluate all images in the directory
for filename in os.listdir(test_images_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for common image file extensions
        file_path = os.path.join(test_images_directory, filename)
        evaluate_image(file_path, discriminator, (IMG_HEIGHT, IMG_WIDTH))

        # After evaluation is done
print(f"Total images evaluated: {results_summary['total_evaluated']}")
print(f"Total real images: {results_summary['total_real']}")
print(f"Total fake images: {results_summary['total_fake']}")
print(f"Total uncertain: {results_summary['uncertain']}")
