import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob

model_dir = './model_storage/'

model_files = glob.glob(f'{model_dir}model_v*.keras')

if model_files:
    model_file = model_files[0]
    print(f"Loading latest model from {model_file}")
    model = load_model(model_file, compile=False)
else:
    print("No model found.")