import os
import tensorflow as tf

model_path = '/volume/'
print(f"Loading model from {model_path}")
# model = tf.keras.models.load_model(model_path)

try:
  content = os.listdir(model_path)
  print(f"Content of {model_path}:")
  for item in content:
      print(item)
except FileNotFoundError:
  print(f"Error: The directory {model_path} does not exist.")
except PermissionError:
  print(f"Error: Permission denied to access {model_path}.")