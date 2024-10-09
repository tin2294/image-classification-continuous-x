# main.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from utils import create_data_generators, create_model

# Constants
BATCH_SIZE = 32
INPUT_IMG_SIZE = 224
TRAINING_DIR = 'data/training/'
EVALUATION_DIR = 'data/evaluation/'
VALIDATION_DIR = 'data/validation/'
N_EPOCHS = 10

# Define class labels
classes = np.array([
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
    "Vegetable/Fruit"
])

# Create data generators
training_gen = create_data_generators(TRAINING_DIR, BATCH_SIZE, INPUT_IMG_SIZE)
validation_gen = create_data_generators(VALIDATION_DIR, BATCH_SIZE, INPUT_IMG_SIZE)
evaluation_gen = create_data_generators(EVALUATION_DIR, BATCH_SIZE, INPUT_IMG_SIZE, is_eval=True)

# Prepare the model
model = create_model(INPUT_IMG_SIZE)

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Train the model
num_training_samples = training_gen.n
num_validation_samples = validation_gen.n

steps_per_epoch = num_training_samples // BATCH_SIZE
validation_steps = num_validation_samples // BATCH_SIZE

hist = model.fit(
    training_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=N_EPOCHS,
    shuffle=True,
    validation_data=validation_gen,
    validation_steps=validation_steps,
    callbacks=[early_stop]
)

# Evaluate the model
eval_results = model.evaluate(evaluation_gen)
print(f'Evaluation results: {eval_results}')

# Make predictions on evaluation data
predictions = model.predict(evaluation_gen)
predicted_classes = np.argmax(predictions, axis=1)

# Print predictions with their corresponding class names
for i in range(len(predicted_classes)):
    print(f'Image: {evaluation_gen.filenames[i]} - Predicted Class: {classes[predicted_classes[i]]}')
