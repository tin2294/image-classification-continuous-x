import numpy as np
import os
import matplotlib.pyplot as plt
from utils import (
    load_training_labels,
    plot_sample_images,
    prepare_data_generators,
    build_model,
    build_transfer_model,
    reorganize_files,
    plot_training_history,
    save_model
)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from modelstore import ModelStore

BATCH_SIZE = 128
INPUT_IMG_SIZE = 112
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

training_images, training_labels = load_training_labels(training_dir)
print(f"Loaded {len(training_images)} training images.")
print(f"Sample training labels: {training_labels[:5]}")

plot_sample_images(CLASSES, training_labels, training_images, n_samples_per_class=4)
print("Sample images plotted.")

# Prepare data generators
training_gen, validation_gen, evaluation_gen = prepare_data_generators(
    training_dir, validation_dir, evaluation_dir, INPUT_IMG_SIZE, BATCH_SIZE)
print(f"Training generator has {training_gen.samples} samples.")
print(f"Validation generator has {validation_gen.samples} samples.")

# Build and train model
# model = build_model(INPUT_IMG_SIZE, len(CLASSES))
model = build_transfer_model(INPUT_IMG_SIZE, len(CLASSES))


# Calculate number of steps
num_training_samples = training_gen.samples
num_validation_samples = validation_gen.samples
# short to make sure workflows are working
# n_epochs = 3
n_epochs = 5
n_epochs_fine = 3

steps_per_epoch = num_training_samples // BATCH_SIZE
validation_steps = num_validation_samples // BATCH_SIZE

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Fit the model
# hist = model.fit(
#     training_gen,
#     steps_per_epoch=steps_per_epoch,
#     epochs=n_epochs,
#     validation_data=validation_gen,
#     validation_steps=validation_steps
# )

hist = model.fit(
    training_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs,
    shuffle=True,
    validation_data=validation_gen,
    validation_steps=validation_steps,
    callbacks=[early_stop]
)

model.trainable = True
for layer in model.layers[:-4]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist_fine = model.fit(
    training_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs_fine,
    shuffle=True,
    validation_data=validation_gen,
    validation_steps=validation_steps,
    callbacks=[early_stop]
)

plot_training_history(hist)
current_dir = os.getcwd()
print("Current working directory:", current_dir)
print("Contents of the directory:")
for item in os.listdir(current_dir):
    item_path = os.path.join(current_dir, item)
    if os.path.isdir(item_path):
        print(f"{item}/ (Directory)")
    elif os.path.isfile(item_path):
        print(f"{item} (File)")
    else:
        print(f"{item} (Other)")
model_storage_dir = "workspace/src/saved_model"
save_model(model, model_storage_dir)

evaluation_loss, evaluation_accuracy = model.evaluate(evaluation_gen, verbose=1)
print("Contents of the directory:")
for item in os.listdir(current_dir):
    item_path = os.path.join(current_dir, item)
    if os.path.isdir(item_path):
        print(f"{item}/ (Directory)")
    elif os.path.isfile(item_path):
        print(f"{item} (File)")
    else:
        print(f"{item} (Other)")

child_directory = "workspace"

# Get the absolute path of the child directory
child_dir_path = os.path.join(current_dir, child_directory)

print("Child directory path:", child_dir_path)

print("Contents of the child directory:")
for item in os.listdir(child_dir_path):
    item_path = os.path.join(child_dir_path, item)
    if os.path.isdir(item_path):
        print(f"{item}/ (Directory)")
    elif os.path.isfile(item_path):
        print(f"{item} (File)")
    else:
        print(f"{item} (Other)")

print(f"Evaluation Loss: {evaluation_loss}", os.getcwd())
print(f"Evaluation Accuracy: {evaluation_accuracy}")

with open("evaluation_metrics.txt", "w") as f:
    f.write(f"evaluation_accuracy: {evaluation_accuracy}\n")
    f.write(f"evaluation_loss: {evaluation_loss}\n")