import tensorflow as tf
import os

MODEL_DIR = "/tmp/model_to_deploy"

keras_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]

if len(keras_files) != 1:
    raise ValueError(f"Expected exactly one .keras file in {MODEL_DIR}, but found {len(keras_files)}.")

MODEL_PATH = os.path.join(MODEL_DIR, keras_files[0])

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")
