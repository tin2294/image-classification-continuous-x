import os
import tensorflow as tf

def create_data_generators(training_dir, validation_dir, batch_size, input_img_size, is_eval=False):
    """Creates image data generators for training and validation datasets."""
    if not is_eval:
        training_aug = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        training_gen = training_aug.flow_from_directory(
            training_dir,
            target_size=(input_img_size, input_img_size),
            color_mode="rgb",
            shuffle=True,
            batch_size=batch_size,
            class_mode='sparse'
        )

        validation_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

        validation_gen = validation_aug.flow_from_directory(
            validation_dir,
            target_size=(input_img_size, input_img_size),
            color_mode="rgb",
            shuffle=True,
            batch_size=batch_size,
            class_mode='sparse'
        )

        return training_gen, validation_gen

    else:
        eval_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

        eval_gen = eval_aug.flow_from_directory(
            training_dir,  # assuming evaluation images are in the training directory
            target_size=(input_img_size, input_img_size),
            color_mode="rgb",
            shuffle=False,
            batch_size=batch_size,
            class_mode='sparse'
        )
        return eval_gen

def create_model(input_img_size):
    """Creates and compiles the image classification model."""
    base_model = tf.keras.applications.VGG16(
        input_shape=(input_img_size, input_img_size, 3),
        include_top=False,
        pooling='avg'
    )
    base_model.trainable = False  # Freeze the base model

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(11, activation='softmax')  # Adjust the number of classes accordingly
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
