import numpy as np
import os
import matplotlib.pyplot as plt
from utils import (
    load_training_labels,
    plot_sample_images,
    prepare_data_generators,
    build_model,
    reorganize_files,
    plot_training_history
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

if not os.path.exists(training_dir):
    raise FileNotFoundError(f"Training directory does not exist: {training_dir}")
if not os.path.exists(validation_dir):
    raise FileNotFoundError(f"Validation directory does not exist: {validation_dir}")
if not os.path.exists(evaluation_dir):
    raise FileNotFoundError(f"Evaluation directory does not exist: {evaluation_dir}")

for dataset in [training_dir, validation_dir, evaluation_dir]:
    reorganize_files(dataset, CLASSES)

print(f"OK")
training_images, training_labels = load_training_labels(training_dir)
print(f"Loaded {len(training_images)} training images.")
print(f"Sample training labels: {training_labels[:5]}")

plot_sample_images(CLASSES, training_labels, training_images, n_samples_per_class=4)
print("Sample images plotted.")

# Prepare data generators
training_gen, validation_gen, evaluation_gen = prepare_data_generators(
    training_dir, validation_dir, evaluation_dir, INPUT_IMG_SIZE, BATCH_SIZE)
print(f"Training generator has {training_gen.samples} samples.")
print(f"Validation generator has {validation_gen.samples} samples.")

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

plot_training_history(hist)
