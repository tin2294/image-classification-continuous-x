import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob
from modelstore import ModelStore

model_dir = './model_storage'

model_store = ModelStore.from_file_system(root_directory=model_dir)
models = model_store.list_versions("image-classification")

for model in models:
  print(model)