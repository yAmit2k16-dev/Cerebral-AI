
import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================
# CEREBRAL AI - MODEL EVALUATION
# ==========================================

MODEL_PATH = r"C:\Brain-Tumor-Detection\best_vgg16_brain_tumor.keras"
TEST_DIR = r"C:\Brain-Tumor-Detection\Testing"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Check paths
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"Testing directory not found: {TEST_DIR}")

# Load trained model
print("Loading Cerebral AI model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load test dataset in deterministic order
print("Loading testing dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

class_names = test_ds.class_names

# Apply the same preprocessing function used in train.py
test_ds = test_ds.map(
    lambda images, labels: (
        tf.keras.applications.vgg16.preprocess_input(images),
        labels
    )
)

# Collect true labels
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

# Predict
print("Generating predictions...")
predictions = model.predict(test_ds, verbose=1)
y_pred = np.argmax(predictions, axis=1)

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 60)
print("       CEREBRAL AI - EVALUATION RESULTS")
print("=" * 60)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Total test images: {len(y_true)}")
print(f"Correct predictions: {np.sum(y_true == y_pred)}")
print(f"Incorrect predictions: {np.sum(y_true != y_pred)}")

# Per-class metrics
print("\nCLASSIFICATION REPORT")
print("-" * 60)

print(classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    digits=4,
    zero_division=0
))

# Confusion matrix
print("CONFUSION MATRIX")
print("-" * 60)
print("Class order:", class_names)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(class_names))
)

print(cm)

# Macro and weighted metrics
print("\nOVERALL METRICS")
print("-" * 60)

for average in ["macro", "weighted"]:
    precision = precision_score(
        y_true, y_pred, average=average, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average=average, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average=average, zero_division=0
    )

    print(f"{average.capitalize()} Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"{average.capitalize()} Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"{average.capitalize()} F1-score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print()

# Save results for research documentation
output_file = r"C:\Brain-Tumor-Detection\evaluation_results.txt"

with open(output_file, "w", encoding="utf-8") as file:
    file.write("CEREBRAL AI - EVALUATION RESULTS\n")
    file.write("=" * 50 + "\n")
    file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Total test images: {len(y_true)}\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write(classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0
    ))

    file.write("\nCONFUSION MATRIX\n")
    file.write(f"Class order: {class_names}\n")
    file.write(np.array2string(cm))

    for average in ["macro", "weighted"]:
        file.write(f"\n\n{average.upper()} METRICS\n")
        file.write(
            f"Precision: {precision_score(y_true, y_pred, average=average, zero_division=0):.4f}\n"
        )
        file.write(
            f"Recall: {recall_score(y_true, y_pred, average=average, zero_division=0):.4f}\n"
        )
        file.write(
            f"F1-score: {f1_score(y_true, y_pred, average=average, zero_division=0):.4f}\n"
        )

print(f"\nResults saved to: {output_file}")
print("\nEvaluation completed.")
