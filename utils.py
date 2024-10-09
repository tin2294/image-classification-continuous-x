import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def load_training_labels(directory):
    """
    Load images and their corresponding labels from the given directory.
    Assumes that images are organized into subdirectories named after their class labels.
    """
    images = []
    labels = []

    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            class_label = int(class_dir.split('_')[1])
            for filename in os.listdir(class_path):
                if filename.lower().endswith('.jpg'):
                    img_path = os.path.join(class_path, filename)
                    images.append(img_path)
                    labels.append(class_label)

    return images, np.array(labels)

def plot_sample_images(classes, labels, image_paths, n_samples_per_class=4):
    """
    Plot sample images from the dataset.
    Displays a fixed number of samples for each class.
    """
    plt.figure(figsize=(15, 10))

    class_samples = {cls: [] for cls in range(len(classes))}
    
    for img_path, label in zip(image_paths, labels):
        label = int(label)  # Ensure label is an integer
        if len(class_samples[label]) < n_samples_per_class:
            class_samples[label].append(img_path)

    for cls, sample_paths in class_samples.items():
        for i, img_path in enumerate(sample_paths):
            img = Image.open(img_path)
            plt.subplot(len(classes), n_samples_per_class, cls * n_samples_per_class + i + 1)
            plt.imshow(img)
            plt.title(classes[cls])
            plt.axis('off')

    plt.tight_layout()
    plt.show()

def create_image_generator(directory, input_size, batch_size):
    """
    Custom data generator for images in a directory with class labels in subdirectory names.
    """
    print("Creating image generator")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    generator = datagen.flow_from_directory(
        directory,
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='sparse'
    )
    
    return generator

def prepare_data_generators(training_dir, validation_dir, evaluation_dir, input_img_size, batch_size):
    """
    Prepares data generators for training, validation, and evaluation datasets.
    """
    training_gen = create_image_generator(training_dir, input_img_size, batch_size)
    validation_gen = create_image_generator(validation_dir, input_img_size, batch_size)
    evaluation_gen = create_image_generator(evaluation_dir, input_img_size, batch_size)

    return training_gen, validation_gen, evaluation_gen

def build_model(input_size, num_classes):
    """
    Build a simple CNN model for image classification.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_size, input_size, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
