"""Evaluate the saved VisionGuard model on CIFAKE's held-out test split.
Run from the visionguard project directory: python evaluate_model.py
"""
import json, os
import numpy as np
import tensorflow as tf
import kagglehub
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, auc)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/image_detector.keras")
OUT_DIR = "evaluation_results"
CLASSES = ["FAKE", "REAL"]  # Must match the training notebook's class order.

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
print("Loading saved model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Finding CIFAKE dataset...")
root = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(root, "test"), class_names=CLASSES, label_mode="binary",
    image_size=(224, 224), batch_size=32, shuffle=False)
print("Predicting test images...")
y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
y_prob = model.predict(test_ds, verbose=1).ravel()
y_pred = (y_prob >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
report = classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASSES,
                               output_dict=True, zero_division=0)
fpr, tpr, _ = roc_curve(y_true, y_prob)
metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "roc_auc": float(auc(fpr, tpr)),
    "test_count": int(len(y_true)),
    "correct": int((y_true == y_pred).sum()),
    "incorrect": int((y_true != y_pred).sum()),
    "confusion_matrix": cm.tolist(),
    "classes": {c: {"precision": float(report[c]["precision"]),
                    "recall": float(report[c]["recall"]),
                    "f1": float(report[c]["f1-score"]),
                    "support": int(report[c]["support"])} for c in CLASSES},
    "roc_curve": [{"fpr": float(x), "tpr": float(y)} for x, y in zip(fpr, tpr)]
}
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASSES, zero_division=0))
print("\nEvaluation complete. Metrics saved to", os.path.join(OUT_DIR, "metrics.json"))
print("Accuracy:", f"{metrics['accuracy']*100:.2f}%", " ROC-AUC:", f"{metrics['roc_auc']:.4f}")
print("Confusion matrix (actual rows FAKE, REAL; predicted columns FAKE, REAL):\n", cm)
