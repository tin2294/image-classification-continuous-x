import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob

volume_dir = '/volume/'
print(f"Contents of {volume_dir}:")
print(os.listdir(volume_dir))

model_files = glob.glob(f'{volume_dir}model_v*.keras')

if model_files:
    model_file = model_files[0]
    print(f"Loading latest model from {model_file}")
    model = load_model(model_file)
else:
    print("No model found.")