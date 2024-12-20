import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import time
from modelstore import ModelStore

def reorganize_files(dataset_path, classes):
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, f"class_{i:02d}")
        os.makedirs(class_dir, exist_ok=True)  # Create class folder if it doesn't exist

        # Find all images for this class in the dataset directory
        files = [f for f in os.listdir(dataset_path) if f.startswith(f"{i}_")]
        for f in files:
            src = os.path.join(dataset_path, f)
            dst = os.path.join(class_dir, f)
            shutil.move(src, dst)

def load_training_labels(directory):
    images = []
    labels = []
    print(f"Inside loader: {directory}")
    print(os.listdir(directory))

    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            class_label = int(filename.split('_')[0])
            images.append(img_path)
            labels.append(class_label)

    print(f"Found {len(images)} images in the directory.")
    return images, np.array(labels)

def plot_sample_images(classes, labels, image_paths, n_samples_per_class=4):
    """
    Plot sample images from the dataset.
    Displays a fixed number of samples for each class.
    """
    plt.figure(figsize=(15, 10))

    class_samples = {cls: [] for cls in range(len(classes))}
    
    for img_path, label in zip(image_paths, labels):
        label = int(label)  # Ensure label is an integer
        if len(class_samples[label]) < n_samples_per_class:
            class_samples[label].append(img_path)

    for cls, sample_paths in class_samples.items():
        for i, img_path in enumerate(sample_paths):
            img = Image.open(img_path)
            plt.subplot(len(classes), n_samples_per_class, cls * n_samples_per_class + i + 1)
            plt.imshow(img)
            plt.title(classes[cls])
            plt.axis('off')

    plt.tight_layout()
    plt.show()

def create_image_generator(directory, input_size, batch_size):
    """
    Custom data generator for images in a directory with class labels in subdirectory names.
    """

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        zoom_range=0.4,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    generator = datagen.flow_from_directory(
        directory,
        target_size=(input_size, input_size),
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size,
        class_mode='sparse'
    )
    
    return generator

def prepare_data_generators(training_dir, validation_dir, evaluation_dir, input_img_size, batch_size):
    """
    Prepares data generators for training, validation, and evaluation datasets.
    """
    training_gen = create_image_generator(training_dir, input_img_size, batch_size)
    validation_gen = create_image_generator(validation_dir, input_img_size, batch_size)
    evaluation_gen = create_image_generator(evaluation_dir, input_img_size, batch_size)

    return training_gen, validation_gen, evaluation_gen

def build_model(input_size, num_classes):
    """
    Build a simple CNN model for image classification.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_size, input_size, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def plot_training_history(hist, filename='training_history.png'):
    """
    Plots the training history of a model.

    Parameters:
        hist: History object containing training history metrics.
        filename: The name of the file to save the plot as.
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['loss'], label='Training Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['accuracy'], label='Training Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plots as images
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_model(model, base_dir):
  storage_path = "/tmp/temp_models/"
  model_store = ModelStore.from_file_system(root_directory=storage_path)
  os.makedirs(base_dir, exist_ok=True)

  timestamp = int(time.time())
  versioned_model_name = f"model_v{timestamp}.keras"
  model_path = os.path.join(base_dir, versioned_model_name)

  model.save(model_path, save_format="keras")
  print(f"Model saved in Keras format at {model_path}")

  result = model_store.upload(
      domain="image-classification",
      model=model_path
  )

  print(f"Model uploaded: {result}")

def plot_confusion_matrix(cm, class_labels, output_path):
  plt.figure(figsize=(10, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion Matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_labels))
  plt.xticks(tick_marks, class_labels, rotation=45)
  plt.yticks(tick_marks, class_labels)

  # Normalize the matrix
  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  thresh = cm.max() / 2.0
  for i, j in np.ndindex(cm.shape):
      plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(output_path)
  plt.close()