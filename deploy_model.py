import os
from modelstore import ModelStore
from tensorflow.keras.models import load_model
import glob

model_file = glob.glob('/volume/model_v*')

if model_file:
    print(f"Loading latest model from {model_file}")
    model = tf.keras.models.load_model(model_file)
else:
    print("No model found.")

