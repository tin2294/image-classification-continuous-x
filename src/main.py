import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from utils import create_data_generators, create_model

# Constants
BATCH_SIZE = 32
INPUT_IMG_SIZE = 224
N_EPOCHS = 10

# Paths to data directories
DATA_DIR = 'Food-11'
TRAINING_DIR = os.path.join(DATA_DIR, 'training')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
EVALUATION_DIR = os.path.join(DATA_DIR, 'evaluation')

# Create data generators
training_gen, validation_gen = create_data_generators(TRAINING_DIR, VALIDATION_DIR, BATCH_SIZE, INPUT_IMG_SIZE)

# Create model
model = create_model(INPUT_IMG_SIZE)

# Model fitting parameters
num_training_samples = training_gen.n
num_validation_samples = validation_gen.n

steps_per_epoch = num_training_samples // BATCH_SIZE
validation_steps = num_validation_samples // BATCH_SIZE

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Train the model
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
evaluation_gen = create_data_generators(EVALUATION_DIR, None, BATCH_SIZE, INPUT_IMG_SIZE, is_eval=True)
eval_results = model.evaluate(evaluation_gen)
print(f'Evaluation results: {eval_results}')
