import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob
from modelstore import ModelStore

model_dir = './model_storage'

model_store = ModelStore.from_file_system(root_directory=model_dir)
models = model_store.list_versions("image-classification")

if len(models) > 0:
  latest_model_id = models[0]
  print(f"Using the latest model with ID: {latest_model_id}")

  clf = model_store.load(
    domain="image-classification",
    model_id=latest_model_id,
  )

  print(f"Model loaded with model store to: {clf}")

  model = load_model(clf, compile=False)
  print("LOADED MODEL WITH TF")

else:
  print("No models found.")
