import numpy as np
import os
from utils import (
    load_training_labels,
    prepare_data_generators,
    build_model,
    save_model
)

BATCH_SIZE = 128
INPUT_IMG_SIZE = 112
CLASSES = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
                    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
                    "Vegetable/Fruit"])

base_dir = "/tmp/content/Food-11"

training_dir = os.path.join(base_dir, "training")
validation_dir = os.path.join(base_dir, "validation")
evaluation_dir = os.path.join(base_dir, "evaluation")

training_images, training_labels = load_training_labels(training_dir)
print(f"Loaded {len(training_images)} training images.")
print(f"Sample training labels: {training_labels[:5]}")

# Prepare data generators
training_gen, validation_gen, evaluation_gen = prepare_data_generators(
    training_dir, validation_dir, evaluation_dir, INPUT_IMG_SIZE, BATCH_SIZE)
print(f"Training generator has {training_gen.samples} samples.")
print(f"Validation generator has {validation_gen.samples} samples.")

# Build and train model
model = build_model(INPUT_IMG_SIZE, len(CLASSES))

# Calculate number of steps
num_training_samples = training_gen.samples
num_validation_samples = validation_gen.samples
# short to make sure workflows are working
n_epochs = 4
# n_epochs = 10
# n_epochs_fine = 3

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

evaluation_loss, evaluation_accuracy = model.evaluate(evaluation_gen, verbose=1)

# Save Model first here then move to ModelStore
model_storage_dir = "./models"
save_model(model, model_storage_dir)

# Create File with Evaluation Metrics
file_path = "/tmp/temp_models/evaluation_metrics.txt"
directory_path = os.path.dirname(file_path)
os.makedirs(directory_path, exist_ok=True)

with open(file_path, "w") as f:
    f.write(f"training_accuracy: {evaluation_accuracy}\n")
    f.write(f"training_loss: {evaluation_loss}\n")

print(f"Training accuracy: {evaluation_accuracy}")
print(f"Training loss: {evaluation_loss}")

print(f"Evaluation metrics written to: {file_path}")
