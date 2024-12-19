import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix, classification_report
from utils import create_image_generator, plot_confusion_matrix

MODEL_DIR = "/tmp/model_to_deploy"
OUTPUT_FILE = "/tmp/temp_models/evaluation_metrics.txt"
DEPLOY = "/tmp/temp_models/deploy.txt"
ACCURACY_THRESHOLD = 0.25
CONFUSION_MATRIX_FILE = "/tmp/temp_models/confusion_matrix.pdf"
EVALUATION_DIR = "/tmp/content/Food-11/evaluation"
INPUT_IMG_SIZE = 112
BATCH_SIZE = 32

# Load Model:
keras_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]

if len(keras_files) != 1:
    raise ValueError(f"Expected exactly one .keras file in {MODEL_DIR}, but found {len(keras_files)}.")

MODEL_PATH = os.path.join(MODEL_DIR, keras_files[0])

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

evaluation_gen = create_image_generator(EVALUATION_DIR, INPUT_IMG_SIZE, BATCH_SIZE)
loss, accuracy = model.evaluate(evaluation_gen, verbose=1)
print(f"Evaluation Loss: {loss:.4f}")
print(f"Evaluation Accuracy: {accuracy:.4f}")

y_pred = model.predict(evaluation_gen)
y_pred_classes = y_pred.argmax(axis=1)
y_true = evaluation_gen.classes

class_labels = list(evaluation_gen.class_indices.keys())

with open(OUTPUT_FILE, "w") as f:
    f.write(f"Evaluation Loss: {loss:.4f}\n")
    f.write(f"Evaluation Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, y_pred_classes, target_names=class_labels))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_true, y_pred_classes)))
print(f"Evaluation metrics saved to {OUTPUT_FILE}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred_classes)
plot_confusion_matrix(cm, class_labels, CONFUSION_MATRIX_FILE)
print(f"Confusion matrix saved to {CONFUSION_MATRIX_FILE}")

print(f"Accuracy: {accuracy}")

deploy = "true" if accuracy >= ACCURACY_THRESHOLD else "false"
print(f"Setting deploy to {deploy}")

with open(DEPLOY, 'w') as f:
    f.write(deploy)