import numpy as np
import os
import matplotlib.pyplot as plt
from utils import (
    load_training_labels,
    plot_sample_images,
    prepare_data_generators,
    build_model,
    # build_transfer_model,
    reorganize_files,
    plot_training_history,
    save_model
)
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
model = build_model(INPUT_IMG_SIZE, len(CLASSES))
# model = build_transfer_model(INPUT_IMG_SIZE, len(CLASSES))


# Calculate number of steps
num_training_samples = training_gen.samples
num_validation_samples = validation_gen.samples
# short to make sure workflows are working
# n_epochs = 2
n_epochs = 10
# n_epochs_fine = 8

steps_per_epoch = num_training_samples // BATCH_SIZE
validation_steps = num_validation_samples // BATCH_SIZE

# early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Fit the model
hist = model.fit(
    training_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs,
    validation_data=validation_gen,
    validation_steps=validation_steps
)

# hist = model.fit(
#     training_gen,
#     steps_per_epoch=steps_per_epoch,
#     epochs=n_epochs,
#     shuffle=True,
#     validation_data=validation_gen,
#     validation_steps=validation_steps,
#     callbacks=[early_stop]
# )

# model.trainable = True
# for layer in model.layers[:-4]:
#     layer.trainable = False

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# hist_fine = model.fit(
#     training_gen,
#     steps_per_epoch=steps_per_epoch,
#     epochs=n_epochs_fine,
#     shuffle=True,
#     validation_data=validation_gen,
#     validation_steps=validation_steps,
#     callbacks=[early_stop]
# )

plot_training_history(hist)

model_storage_dir = "saved_model"
save_model(model, model_storage_dir)

evaluation_loss, evaluation_accuracy = model.evaluate(evaluation_gen, verbose=1)
print(f"Evaluation Loss: {evaluation_loss}")
print(f"Evaluation Accuracy: {evaluation_accuracy}")

with open("evaluation_metrics.txt", "w") as f:
    f.write(f"evaluation_accuracy: {evaluation_accuracy}\n")
    f.write(f"evaluation_loss: {evaluation_loss}\n")