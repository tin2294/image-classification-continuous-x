import numpy as np
import os
import matplotlib.pyplot as plt
from utils import (
    load_training_labels,
    plot_sample_images,
    prepare_data_generators,
    build_model
)

BATCH_SIZE = 128
INPUT_IMG_SIZE = 112
CLASSES = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
                    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
                    "Vegetable/Fruit"])

base_dir = "content/Food-11"

training_dir = os.path.join(base_dir, "training")
validation_dir = os.path.join(base_dir, "validation")
evaluation_dir = os.path.join(base_dir, "evaluation")

training_images, training_labels = load_training_labels(training_dir)

plot_sample_images(CLASSES, training_labels, training_images, n_samples_per_class=4)

# Prepare data generators
training_gen, validation_gen, evaluation_gen = prepare_data_generators(
    training_dir, validation_dir, evaluation_dir, INPUT_IMG_SIZE, BATCH_SIZE)

# Build and train model
model = build_model(INPUT_IMG_SIZE, len(CLASSES))

num_training_samples = sum(len(files) for _, _, files in os.walk(training_dir))  # Total number of training images
num_validation_samples = sum(len(files) for _, _, files in os.walk(validation_dir))  # Total number of validation images
n_epochs = 10

steps_per_epoch = num_training_samples // BATCH_SIZE
validation_steps = num_validation_samples // BATCH_SIZE

# Fit the model
hist = model.fit(
    training_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs,
    validation_data=validation_gen,
    validation_steps=validation_steps
)

# Plot training history
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
