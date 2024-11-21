import os
import reorganize_files from utils

CLASSES = np.array(["Bread", "Dairy product", "Dessert", "Egg", "Fried food",
                    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
                    "Vegetable/Fruit"])

base_dir = "/tmp/content/Food-11"

training_dir = os.path.join(base_dir, "training")
validation_dir = os.path.join(base_dir, "validation")
evaluation_dir = os.path.join(base_dir, "evaluation")

for dataset in [training_dir, validation_dir, evaluation_dir]:
  if not os.path.exists(dataset):
    raise FileNotFoundError(f"Directory does not exist: {dataset}")

for dataset in [training_dir, validation_dir, evaluation_dir]:
  reorganize_files(dataset, CLASSES)