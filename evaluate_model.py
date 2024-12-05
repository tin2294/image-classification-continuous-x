import tensorflow as tf
import numpy as np
import json
import os

# Paths
MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/temp_models/latest_model.keras")
DATASET_PATH = os.getenv("DATASET_PATH", "/tmp/content/Food-11/validation")
METRICS_OUTPUT_PATH = os.getenv("METRICS_OUTPUT_PATH", "/tmp/temp_models/evaluation_metrics.txt")

# Load the model
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")
