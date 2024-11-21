import os
from modelstore import ModelStore
from tensorflow.keras.models import load_model

storage_path = "/volume/"
model_store = ModelStore.from_file_system(root_directory=storage_path)
models = model_store.list_versions("image-classification")

if len(models) > 0:
  latest_model_id = models[0]
  print(f"Using the latest model with ID: {latest_model_id}")

  local_download_dir = "/volume/downloaded_model"
  os.makedirs(local_download_dir, exist_ok=True)

  model_file_path = model_store.download(
      domain="image-classification",
      model_id=latest_model_id,
      local_path=local_download_dir
  )

  print(f"Model downloaded to: {model_file_path}")

  full_model_path = os.path.join(model_file_path, versioned_model_name)

  try:
      model = load_model(full_model_path)
      print("Model loaded successfully.")
  except Exception as e:
      print(f"Error loading model: {e}")
else:
    print("No models found.")
